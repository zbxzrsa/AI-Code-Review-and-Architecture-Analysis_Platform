"""
Mutation Testing Framework (R-003 Mitigation)

Implements mutation testing to verify test quality beyond coverage metrics.

Features:
- Automated mutant generation
- Test suite execution against mutants
- Mutation score calculation
- Surviving mutant analysis
- CI/CD integration

Target: Mutation score >= 70%
"""
import ast
import copy
import importlib
import logging
import os
import random
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import json

logger = logging.getLogger(__name__)


class MutationType(str, Enum):
    """Types of mutations."""
    # Arithmetic operators
    ARITHMETIC = "arithmetic"           # + -> -, * -> /
    # Comparison operators
    COMPARISON = "comparison"           # == -> !=, < -> >=
    # Logical operators
    LOGICAL = "logical"                 # and -> or
    # Boolean literals
    BOOLEAN = "boolean"                 # True -> False
    # Number literals
    NUMBER = "number"                   # n -> n+1, n-1
    # Return values
    RETURN = "return"                   # return x -> return None
    # Conditionals
    CONDITIONAL = "conditional"         # if x -> if not x
    # Boundary
    BOUNDARY = "boundary"               # < -> <=, > -> >=
    # Remove statement
    STATEMENT_DELETION = "statement_deletion"


class MutantStatus(str, Enum):
    """Status of a mutant."""
    KILLED = "killed"           # Tests detected the mutation
    SURVIVED = "survived"       # Tests did not detect the mutation
    TIMEOUT = "timeout"         # Tests timed out
    ERROR = "error"             # Error running tests
    EQUIVALENT = "equivalent"   # Equivalent mutant (same behavior)


@dataclass
class Mutant:
    """Represents a code mutation."""
    id: str
    file_path: str
    line_number: int
    mutation_type: MutationType
    original_code: str
    mutated_code: str
    status: MutantStatus = MutantStatus.SURVIVED
    test_output: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "mutation_type": self.mutation_type.value,
            "original_code": self.original_code,
            "mutated_code": self.mutated_code,
            "status": self.status.value,
        }


@dataclass
class MutationResult:
    """Result of mutation testing."""
    total_mutants: int
    killed: int
    survived: int
    timeout: int
    errors: int
    equivalent: int
    mutants: List[Mutant]
    duration_seconds: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    @property
    def mutation_score(self) -> float:
        """Calculate mutation score percentage."""
        effective_mutants = self.total_mutants - self.equivalent
        if effective_mutants == 0:
            return 100.0
        return (self.killed / effective_mutants) * 100
    
    @property
    def passed(self) -> bool:
        """Check if mutation score meets threshold (70%)."""
        return self.mutation_score >= 70.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": {
                "total_mutants": self.total_mutants,
                "killed": self.killed,
                "survived": self.survived,
                "timeout": self.timeout,
                "errors": self.errors,
                "equivalent": self.equivalent,
                "mutation_score": round(self.mutation_score, 2),
                "passed": self.passed,
                "duration_seconds": round(self.duration_seconds, 2),
                "timestamp": self.timestamp.isoformat(),
            },
            "surviving_mutants": [
                m.to_dict() for m in self.mutants 
                if m.status == MutantStatus.SURVIVED
            ],
        }


class MutationOperator(ast.NodeTransformer):
    """
    AST transformer that applies mutations to Python code.
    """
    
    # Arithmetic operator replacements
    ARITHMETIC_OPS = {
        ast.Add: [ast.Sub, ast.Mult],
        ast.Sub: [ast.Add, ast.Mult],
        ast.Mult: [ast.Div, ast.Add],
        ast.Div: [ast.Mult, ast.Sub],
        ast.Mod: [ast.Div, ast.Mult],
    }
    
    # Comparison operator replacements
    COMPARISON_OPS = {
        ast.Eq: [ast.NotEq],
        ast.NotEq: [ast.Eq],
        ast.Lt: [ast.LtE, ast.Gt, ast.GtE],
        ast.LtE: [ast.Lt, ast.Gt, ast.GtE],
        ast.Gt: [ast.GtE, ast.Lt, ast.LtE],
        ast.GtE: [ast.Gt, ast.Lt, ast.LtE],
    }
    
    # Logical operator replacements
    LOGICAL_OPS = {
        ast.And: [ast.Or],
        ast.Or: [ast.And],
    }
    
    def __init__(self, mutation_type: MutationType, target_line: Optional[int] = None):
        self.mutation_type = mutation_type
        self.target_line = target_line
        self.mutations_applied = []
    
    def visit_BinOp(self, node):
        """Mutate binary operations."""
        if self.mutation_type == MutationType.ARITHMETIC:
            if self.target_line is None or node.lineno == self.target_line:
                op_type = type(node.op)
                if op_type in self.ARITHMETIC_OPS:
                    replacements = self.ARITHMETIC_OPS[op_type]
                    if replacements:
                        new_op = random.choice(replacements)()
                        self.mutations_applied.append({
                            "line": node.lineno,
                            "original": op_type.__name__,
                            "mutated": new_op.__class__.__name__,
                        })
                        node.op = new_op
        
        return self.generic_visit(node)
    
    def visit_Compare(self, node):
        """Mutate comparison operations."""
        if self.mutation_type in [MutationType.COMPARISON, MutationType.BOUNDARY]:
            if self.target_line is None or node.lineno == self.target_line:
                new_ops = []
                for op in node.ops:
                    op_type = type(op)
                    if op_type in self.COMPARISON_OPS:
                        replacements = self.COMPARISON_OPS[op_type]
                        if replacements:
                            new_op = random.choice(replacements)()
                            self.mutations_applied.append({
                                "line": node.lineno,
                                "original": op_type.__name__,
                                "mutated": new_op.__class__.__name__,
                            })
                            new_ops.append(new_op)
                        else:
                            new_ops.append(op)
                    else:
                        new_ops.append(op)
                node.ops = new_ops
        
        return self.generic_visit(node)
    
    def visit_BoolOp(self, node):
        """Mutate boolean operations."""
        if self.mutation_type == MutationType.LOGICAL:
            if self.target_line is None or node.lineno == self.target_line:
                op_type = type(node.op)
                if op_type in self.LOGICAL_OPS:
                    replacements = self.LOGICAL_OPS[op_type]
                    if replacements:
                        new_op = random.choice(replacements)()
                        self.mutations_applied.append({
                            "line": node.lineno,
                            "original": op_type.__name__,
                            "mutated": new_op.__class__.__name__,
                        })
                        node.op = new_op
        
        return self.generic_visit(node)
    
    def visit_NameConstant(self, node):
        """Mutate boolean constants (Python < 3.8)."""
        return self._mutate_constant(node)
    
    def visit_Constant(self, node):
        """Mutate constants (Python >= 3.8)."""
        return self._mutate_constant(node)
    
    def _mutate_constant(self, node):
        """Mutate constant values."""
        if self.mutation_type == MutationType.BOOLEAN:
            if self.target_line is None or node.lineno == self.target_line:
                if isinstance(node.value, bool):
                    self.mutations_applied.append({
                        "line": node.lineno,
                        "original": str(node.value),
                        "mutated": str(not node.value),
                    })
                    node.value = not node.value
        
        elif self.mutation_type == MutationType.NUMBER:
            if self.target_line is None or node.lineno == self.target_line:
                if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                    original = node.value
                    mutations = [original + 1, original - 1, 0, -original]
                    new_value = random.choice([m for m in mutations if m != original])
                    self.mutations_applied.append({
                        "line": node.lineno,
                        "original": str(original),
                        "mutated": str(new_value),
                    })
                    node.value = new_value
        
        return node
    
    def visit_Return(self, node):
        """Mutate return statements."""
        if self.mutation_type == MutationType.RETURN:
            if self.target_line is None or node.lineno == self.target_line:
                if node.value is not None:
                    self.mutations_applied.append({
                        "line": node.lineno,
                        "original": "return <value>",
                        "mutated": "return None",
                    })
                    node.value = ast.Constant(value=None)
        
        return self.generic_visit(node)
    
    def visit_If(self, node):
        """Mutate conditionals."""
        if self.mutation_type == MutationType.CONDITIONAL:
            if self.target_line is None or node.lineno == self.target_line:
                # Negate the condition
                self.mutations_applied.append({
                    "line": node.lineno,
                    "original": "if <condition>",
                    "mutated": "if not <condition>",
                })
                node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        
        return self.generic_visit(node)


class MutationTester:
    """
    Main mutation testing engine.
    
    Features:
    - Generate mutants from source code
    - Run tests against mutants
    - Calculate mutation score
    - Generate reports
    """
    
    def __init__(
        self,
        source_dirs: List[str],
        test_command: str = "pytest",
        timeout: int = 60,
        max_mutants: int = 100,
    ):
        self.source_dirs = [Path(d) for d in source_dirs]
        self.test_command = test_command
        self.timeout = timeout
        self.max_mutants = max_mutants
        self.mutants: List[Mutant] = []
    
    def run(self) -> MutationResult:
        """Run mutation testing."""
        import time
        start_time = time.time()
        
        logger.info("Starting mutation testing...")
        
        # 1. Generate mutants
        self._generate_mutants()
        
        # 2. Run tests against each mutant
        for mutant in self.mutants:
            self._test_mutant(mutant)
        
        # 3. Calculate results
        duration = time.time() - start_time
        
        result = MutationResult(
            total_mutants=len(self.mutants),
            killed=sum(1 for m in self.mutants if m.status == MutantStatus.KILLED),
            survived=sum(1 for m in self.mutants if m.status == MutantStatus.SURVIVED),
            timeout=sum(1 for m in self.mutants if m.status == MutantStatus.TIMEOUT),
            errors=sum(1 for m in self.mutants if m.status == MutantStatus.ERROR),
            equivalent=sum(1 for m in self.mutants if m.status == MutantStatus.EQUIVALENT),
            mutants=self.mutants,
            duration_seconds=duration,
        )
        
        logger.info(f"Mutation testing complete: {result.mutation_score:.1f}% mutation score")
        
        return result
    
    def _generate_mutants(self):
        """Generate mutants from source files."""
        mutant_count = 0
        
        for source_dir in self.source_dirs:
            for py_file in source_dir.rglob("*.py"):
                if mutant_count >= self.max_mutants:
                    break
                
                # Skip test files and __pycache__
                if "test" in py_file.name.lower() or "__pycache__" in str(py_file):
                    continue
                
                try:
                    mutants = self._generate_mutants_for_file(py_file)
                    for mutant in mutants:
                        if mutant_count >= self.max_mutants:
                            break
                        self.mutants.append(mutant)
                        mutant_count += 1
                except Exception as e:
                    logger.warning(f"Failed to generate mutants for {py_file}: {e}")
        
        logger.info(f"Generated {len(self.mutants)} mutants")
    
    def _generate_mutants_for_file(self, file_path: Path) -> List[Mutant]:
        """Generate mutants for a single file."""
        with open(file_path) as f:
            source = f.read()
        
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []
        
        mutants = []
        mutation_types = list(MutationType)
        
        # Find mutable lines
        for node in ast.walk(tree):
            if not hasattr(node, "lineno"):
                continue
            
            line_no = node.lineno
            
            # Try different mutation types
            for mutation_type in mutation_types:
                try:
                    tree_copy = ast.parse(source)
                    operator = MutationOperator(mutation_type, line_no)
                    mutated_tree = operator.visit(tree_copy)
                    
                    if operator.mutations_applied:
                        # Fix missing line numbers
                        ast.fix_missing_locations(mutated_tree)
                        
                        try:
                            mutated_code = ast.unparse(mutated_tree)
                        except Exception:
                            continue
                        
                        # Get original line
                        lines = source.splitlines()
                        original_line = lines[line_no - 1] if line_no <= len(lines) else ""
                        
                        mutant = Mutant(
                            id=f"M-{len(mutants):04d}",
                            file_path=str(file_path),
                            line_number=line_no,
                            mutation_type=mutation_type,
                            original_code=original_line.strip(),
                            mutated_code=str(operator.mutations_applied[0]),
                        )
                        mutants.append(mutant)
                        
                        if len(mutants) >= 10:  # Limit per file
                            return mutants
                            
                except Exception:
                    continue
        
        return mutants
    
    def _test_mutant(self, mutant: Mutant):
        """Test a single mutant."""
        file_path = Path(mutant.file_path)
        
        # Read original code
        with open(file_path) as f:
            original_source = f.read()
        
        try:
            # Parse and mutate
            tree = ast.parse(original_source)
            operator = MutationOperator(mutant.mutation_type, mutant.line_number)
            mutated_tree = operator.visit(tree)
            ast.fix_missing_locations(mutated_tree)
            
            try:
                mutated_source = ast.unparse(mutated_tree)
            except Exception:
                mutant.status = MutantStatus.ERROR
                return
            
            # Write mutated code
            with open(file_path, "w") as f:
                f.write(mutated_source)
            
            # Run tests
            try:
                result = subprocess.run(
                    self.test_command.split(),
                    capture_output=True,
                    timeout=self.timeout,
                    text=True,
                )
                
                if result.returncode != 0:
                    mutant.status = MutantStatus.KILLED
                    mutant.test_output = result.stdout[:500]
                else:
                    mutant.status = MutantStatus.SURVIVED
                    
            except subprocess.TimeoutExpired:
                mutant.status = MutantStatus.TIMEOUT
            except Exception as e:
                mutant.status = MutantStatus.ERROR
                mutant.test_output = str(e)
                
        finally:
            # Restore original code
            with open(file_path, "w") as f:
                f.write(original_source)
    
    def generate_report(self, result: MutationResult, output_path: str):
        """Generate mutation testing report."""
        report = result.to_dict()
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        # Generate HTML report
        html_path = output_path.replace(".json", ".html")
        self._generate_html_report(result, html_path)
        
        logger.info(f"Report saved: {output_path}")
    
    def _generate_html_report(self, result: MutationResult, output_path: str):
        """Generate HTML mutation report."""
        surviving = [m for m in result.mutants if m.status == MutantStatus.SURVIVED]
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Mutation Testing Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .summary {{ display: flex; gap: 20px; margin-bottom: 20px; }}
        .card {{ background: #f5f5f5; padding: 15px; border-radius: 8px; text-align: center; }}
        .card.passed {{ border-top: 4px solid #4caf50; }}
        .card.failed {{ border-top: 4px solid #f44336; }}
        .score {{ font-size: 2em; font-weight: bold; }}
        .score.passed {{ color: #4caf50; }}
        .score.failed {{ color: #f44336; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #f5f5f5; }}
        .status-killed {{ color: #4caf50; }}
        .status-survived {{ color: #f44336; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
    </style>
</head>
<body>
    <h1>Mutation Testing Report</h1>
    <p>Generated: {result.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <div class="summary">
        <div class="card {'passed' if result.passed else 'failed'}">
            <div class="score {'passed' if result.passed else 'failed'}">{result.mutation_score:.1f}%</div>
            <div>Mutation Score</div>
            <div>{'✅ PASSED' if result.passed else '❌ FAILED'} (Target: 70%)</div>
        </div>
        <div class="card">
            <div class="score">{result.killed}</div>
            <div>Killed</div>
        </div>
        <div class="card">
            <div class="score">{result.survived}</div>
            <div>Survived</div>
        </div>
        <div class="card">
            <div class="score">{result.total_mutants}</div>
            <div>Total Mutants</div>
        </div>
    </div>
    
    <h2>Surviving Mutants</h2>
    <p>These mutants were not detected by tests. Consider adding tests for these cases.</p>
    
    <table>
        <tr>
            <th>ID</th>
            <th>File</th>
            <th>Line</th>
            <th>Type</th>
            <th>Original</th>
            <th>Mutation</th>
        </tr>
        {''.join(f'''
        <tr>
            <td>{m.id}</td>
            <td><code>{Path(m.file_path).name}</code></td>
            <td>{m.line_number}</td>
            <td>{m.mutation_type.value}</td>
            <td><code>{m.original_code[:50]}</code></td>
            <td><code>{m.mutated_code}</code></td>
        </tr>
        ''' for m in surviving[:20])}
    </table>
    
    <h2>Statistics</h2>
    <ul>
        <li>Duration: {result.duration_seconds:.1f} seconds</li>
        <li>Killed: {result.killed} ({result.killed/result.total_mutants*100:.1f}%)</li>
        <li>Survived: {result.survived} ({result.survived/result.total_mutants*100:.1f}%)</li>
        <li>Timeout: {result.timeout}</li>
        <li>Errors: {result.errors}</li>
    </ul>
</body>
</html>"""
        
        with open(output_path, "w") as f:
            f.write(html)


# CLI entry point
def main():
    """CLI entry point for mutation testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run mutation testing")
    parser.add_argument("--source", "-s", nargs="+", default=["backend"], help="Source directories")
    parser.add_argument("--test-command", "-t", default="pytest tests/unit -x -q", help="Test command")
    parser.add_argument("--timeout", type=int, default=30, help="Test timeout per mutant")
    parser.add_argument("--max-mutants", "-m", type=int, default=50, help="Maximum mutants to generate")
    parser.add_argument("--output", "-o", default="mutation_report.json", help="Output report path")
    
    args = parser.parse_args()
    
    tester = MutationTester(
        source_dirs=args.source,
        test_command=args.test_command,
        timeout=args.timeout,
        max_mutants=args.max_mutants,
    )
    
    result = tester.run()
    tester.generate_report(result, args.output)
    
    print(f"\n{'='*50}")
    print(f"Mutation Score: {result.mutation_score:.1f}%")
    print(f"Status: {'PASSED ✅' if result.passed else 'FAILED ❌'}")
    print(f"{'='*50}")
    
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
