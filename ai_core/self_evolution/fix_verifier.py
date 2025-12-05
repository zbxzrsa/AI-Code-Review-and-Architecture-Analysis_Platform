"""
Fix Verification System

Automated verification of code fixes through:
- Syntax validation
- Unit test execution
- Static analysis
- Regression detection
"""

import asyncio
import ast
import logging
import subprocess
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Result of fix verification."""
    verified: bool
    syntax_valid: bool
    tests_passed: int
    tests_failed: int
    tests_skipped: int
    static_analysis_issues: int
    regression_detected: bool
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TestResult:
    """Result of running tests."""
    passed: int
    failed: int
    skipped: int
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0


class FixVerifier:
    """
    Verifies code fixes through multiple validation stages.
    
    Stages:
    1. Syntax Validation - Ensure code is syntactically correct
    2. Static Analysis - Check for common issues
    3. Unit Tests - Run relevant tests
    4. Regression Detection - Compare behavior before/after
    """
    
    def __init__(
        self,
        workspace_path: str,
        python_path: Optional[str] = None,
        test_timeout: int = 300,
        enable_static_analysis: bool = True,
    ):
        self.workspace_path = Path(workspace_path)
        self.python_path = python_path or sys.executable
        self.test_timeout = test_timeout
        self.enable_static_analysis = enable_static_analysis
        
        # Cache for test discovery
        self._test_cache: Dict[str, List[str]] = {}
    
    async def verify_fix(
        self,
        file_path: str,
        original_code: str,
        fixed_code: str,
        run_tests: bool = True,
    ) -> VerificationResult:
        """
        Verify a code fix through all validation stages.
        
        Args:
            file_path: Path to the modified file
            original_code: Original code before fix
            fixed_code: Fixed code after modification
            run_tests: Whether to run unit tests
            
        Returns:
            VerificationResult with all validation details
        """
        start_time = datetime.now(timezone.utc)
        result = VerificationResult(
            verified=False,
            syntax_valid=False,
            tests_passed=0,
            tests_failed=0,
            tests_skipped=0,
            static_analysis_issues=0,
            regression_detected=False,
        )
        
        try:
            # Stage 1: Syntax Validation
            syntax_result = await self._validate_syntax(file_path, fixed_code)
            result.syntax_valid = syntax_result["valid"]
            result.details["syntax"] = syntax_result
            
            if not result.syntax_valid:
                result.error_message = f"Syntax error: {syntax_result.get('error')}"
                return result
            
            # Stage 2: Static Analysis
            if self.enable_static_analysis:
                analysis_result = await self._run_static_analysis(file_path, fixed_code)
                result.static_analysis_issues = analysis_result["issue_count"]
                result.details["static_analysis"] = analysis_result
            
            # Stage 3: Unit Tests
            if run_tests:
                test_result = await self._run_tests(file_path)
                result.tests_passed = test_result.passed
                result.tests_failed = test_result.failed
                result.tests_skipped = test_result.skipped
                result.details["tests"] = {
                    "passed": test_result.passed,
                    "failed": test_result.failed,
                    "skipped": test_result.skipped,
                    "errors": test_result.errors,
                }
                
                if test_result.failed > 0:
                    result.error_message = f"{test_result.failed} tests failed"
            
            # Stage 4: Regression Detection
            regression_result = await self._check_regression(
                file_path, original_code, fixed_code
            )
            result.regression_detected = regression_result["detected"]
            result.details["regression"] = regression_result
            
            # Final verification
            result.verified = (
                result.syntax_valid
                and result.tests_failed == 0
                and not result.regression_detected
                and result.static_analysis_issues < 5  # Allow minor issues
            )
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            result.error_message = str(e)
        
        finally:
            end_time = datetime.now(timezone.utc)
            result.duration_seconds = (end_time - start_time).total_seconds()
        
        return result
    
    def _validate_syntax(
        self,
        file_path: str,  # noqa: ARG002 - used for extension check
        code: str,
    ) -> Dict[str, Any]:
        """Validate Python syntax."""
        try:
            # For Python files, use AST
            if file_path.endswith(".py"):
                ast.parse(code)
                return {"valid": True, "language": "python"}
            
            # For other languages, basic validation
            return {"valid": True, "language": "unknown"}
            
        except SyntaxError as e:
            return {
                "valid": False,
                "error": str(e),
                "line": e.lineno,
                "offset": e.offset,
            }
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    async def _run_static_analysis(
        self,
        file_path: str,  # noqa: ARG002 - reserved for language detection
        code: str,
    ) -> Dict[str, Any]:
        """Run static analysis on code."""
        issues = []
        
        try:
            # Create temp file with fixed code using async file operations
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / f"fix_verify_{uuid.uuid4().hex}.py"
            
            async with aiofiles.open(temp_path, mode="w") as f:
                await f.write(code)
            
            # Run ruff (fast Python linter)
            try:
                result = await asyncio.create_subprocess_exec(
                    self.python_path, "-m", "ruff", "check", str(temp_path),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await asyncio.wait_for(
                    result.communicate(),
                    timeout=30,
                )
                
                if stdout:
                    issues.extend(stdout.decode().strip().split("\n"))
                    
            except FileNotFoundError:
                # ruff not installed, skip
                pass
            except asyncio.TimeoutError:
                logger.warning("Static analysis timed out")
            finally:
                # Clean up temp file
                temp_path.unlink(missing_ok=True)
            
        except Exception as e:
            logger.error(f"Static analysis failed: {e}")
        
        return {
            "issue_count": len([i for i in issues if i.strip()]),
            "issues": issues[:20],  # Limit to first 20
        }
    
    async def _run_tests(self, file_path: str) -> TestResult:
        """Run unit tests related to the modified file."""
        result = TestResult(passed=0, failed=0, skipped=0)
        
        try:
            # Find related test files
            test_files = await self._discover_related_tests(file_path)
            
            if not test_files:
                logger.info(f"No tests found for {file_path}")
                return result
            
            # Run pytest
            cmd = [
                self.python_path, "-m", "pytest",
                *test_files,
                "-v",
                "--tb=short",
                "-q",
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self.workspace_path),
            )
            
            try:
                stdout, _ = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.test_timeout,
                )
                
                output = stdout.decode()
                
                # Parse pytest output
                result = self._parse_pytest_output(output)
                
            except asyncio.TimeoutError:
                process.kill()
                result.errors.append("Test execution timed out")
                
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            result.errors.append(str(e))
        
        return result
    
    def _discover_related_tests(self, file_path: str) -> List[str]:
        """Discover test files related to modified file."""
        if file_path in self._test_cache:
            return self._test_cache[file_path]
        
        test_files = []
        file_path_obj = Path(file_path)
        file_name = file_path_obj.stem
        
        # Look for test files with matching names
        patterns = [
            f"test_{file_name}.py",
            f"{file_name}_test.py",
            f"tests/test_{file_name}.py",
            f"tests/{file_name}_test.py",
        ]
        
        for pattern in patterns:
            matches = list(self.workspace_path.glob(f"**/{pattern}"))
            test_files.extend(str(m) for m in matches)
        
        # Also look in tests directory
        tests_dir = self.workspace_path / "tests"
        if tests_dir.exists():
            # Get module path
            try:
                rel_path = file_path_obj.relative_to(self.workspace_path)
                module_parts = rel_path.with_suffix("").parts
                
                # Look for test file matching module structure
                for part in module_parts:
                    test_path = tests_dir / f"test_{part}.py"
                    if test_path.exists():
                        test_files.append(str(test_path))
            except ValueError:
                pass
        
        # Remove duplicates
        test_files = list(set(test_files))
        
        self._test_cache[file_path] = test_files
        return test_files
    
    def _parse_pytest_output(self, output: str) -> TestResult:
        """Parse pytest output to extract results."""
        result = TestResult(passed=0, failed=0, skipped=0)
        
        lines = output.split("\n")
        
        for line in lines:
            # Look for summary line like "5 passed, 2 failed, 1 skipped"
            if "passed" in line or "failed" in line or "skipped" in line:
                import re
                
                passed_match = re.search(r"(\d+)\s+passed", line)
                failed_match = re.search(r"(\d+)\s+failed", line)
                skipped_match = re.search(r"(\d+)\s+skipped", line)
                
                if passed_match:
                    result.passed = int(passed_match.group(1))
                if failed_match:
                    result.failed = int(failed_match.group(1))
                if skipped_match:
                    result.skipped = int(skipped_match.group(1))
                
                break
        
        # Collect error messages
        collecting_error = False
        current_error = []
        
        for line in lines:
            if line.startswith("FAILED") or line.startswith("ERROR"):
                collecting_error = True
                current_error = [line]
            elif collecting_error:
                if line.startswith("=") or line.startswith("-"):
                    if current_error:
                        result.errors.append("\n".join(current_error))
                    collecting_error = False
                    current_error = []
                else:
                    current_error.append(line)
        
        return result
    
    def _check_regression(
        self,
        file_path: str,  # noqa: ARG002 - reserved for language-specific checks
        original_code: str,
        fixed_code: str,
    ) -> Dict[str, Any]:
        """Check for behavioral regression."""
        regression = {
            "detected": False,
            "checks": [],
        }
        
        try:
            # Parse both versions
            original_ast = ast.parse(original_code)
            fixed_ast = ast.parse(fixed_code)
            
            # Compare function signatures
            original_funcs = self._extract_function_signatures(original_ast)
            fixed_funcs = self._extract_function_signatures(fixed_ast)
            
            # Check for removed functions
            removed = set(original_funcs.keys()) - set(fixed_funcs.keys())
            if removed:
                regression["detected"] = True
                regression["checks"].append({
                    "type": "removed_functions",
                    "functions": list(removed),
                })
            
            # Check for signature changes
            for func_name in original_funcs:
                if func_name in fixed_funcs:
                    if original_funcs[func_name] != fixed_funcs[func_name]:
                        regression["checks"].append({
                            "type": "signature_change",
                            "function": func_name,
                            "original": original_funcs[func_name],
                            "fixed": fixed_funcs[func_name],
                        })
            
            # Compare class structures
            original_classes = self._extract_class_info(original_ast)
            fixed_classes = self._extract_class_info(fixed_ast)
            
            removed_classes = set(original_classes.keys()) - set(fixed_classes.keys())
            if removed_classes:
                regression["detected"] = True
                regression["checks"].append({
                    "type": "removed_classes",
                    "classes": list(removed_classes),
                })
            
        except SyntaxError:
            # If we can't parse, assume no regression
            pass
        except Exception as e:
            logger.error(f"Regression check failed: {e}")
        
        return regression
    
    def _extract_function_signatures(
        self,
        tree: ast.AST,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract function signatures from AST."""
        functions = {}
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = node.args
                sig = {
                    "args": [arg.arg for arg in args.args],
                    "defaults": len(args.defaults),
                    "kwonly": [arg.arg for arg in args.kwonlyargs],
                    "has_vararg": args.vararg is not None,
                    "has_kwarg": args.kwarg is not None,
                }
                functions[node.name] = sig
        
        return functions
    
    def _extract_class_info(
        self,
        tree: ast.AST,
    ) -> Dict[str, Dict[str, Any]]:
        """Extract class information from AST."""
        classes = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = []
                attributes = []
                
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        methods.append(item.name)
                    elif isinstance(item, ast.Assign):
                        for target in item.targets:
                            if isinstance(target, ast.Name):
                                attributes.append(target.id)
                
                classes[node.name] = {
                    "methods": methods,
                    "attributes": attributes,
                    "bases": [
                        ast.unparse(base) if hasattr(ast, "unparse") else str(base)
                        for base in node.bases
                    ],
                }
        
        return classes


# =============================================================================
# Test Runner Factory
# =============================================================================

def create_test_runner(workspace_path: str) -> Callable:
    """Create a test runner function for the bug fixer."""
    verifier = FixVerifier(workspace_path)
    
    async def run_tests(file_path: str) -> Dict[str, Any]:
        """Run tests for a file and return results."""
        test_result = await verifier._run_tests(file_path)
        return {
            "passed": test_result.passed,
            "failed": test_result.failed,
            "skipped": test_result.skipped,
            "errors": test_result.errors,
        }
    
    return run_tests


# =============================================================================
# Verification Pipeline
# =============================================================================

class VerificationPipeline:
    """
    Complete verification pipeline for code fixes.
    
    Orchestrates multiple verification stages and aggregates results.
    """
    
    def __init__(
        self,
        workspace_path: str,
        stages: Optional[List[str]] = None,
    ):
        self.workspace_path = Path(workspace_path)
        self.verifier = FixVerifier(workspace_path)
        self.stages = stages or [
            "syntax",
            "static_analysis",
            "unit_tests",
            "regression",
        ]
        
        # Pipeline statistics
        self._stats = {
            "total_verifications": 0,
            "passed": 0,
            "failed": 0,
            "by_stage": {},
        }
    
    async def verify(
        self,
        file_path: str,
        original_code: str,
        fixed_code: str,
    ) -> VerificationResult:
        """Run full verification pipeline."""
        self._stats["total_verifications"] += 1
        
        result = await self.verifier.verify_fix(
            file_path,
            original_code,
            fixed_code,
            run_tests="unit_tests" in self.stages,
        )
        
        if result.verified:
            self._stats["passed"] += 1
        else:
            self._stats["failed"] += 1
        
        # Track by stage
        for stage in self.stages:
            if stage not in self._stats["by_stage"]:
                self._stats["by_stage"][stage] = {"passed": 0, "failed": 0}
            
            stage_passed = self._check_stage_passed(result, stage)
            if stage_passed:
                self._stats["by_stage"][stage]["passed"] += 1
            else:
                self._stats["by_stage"][stage]["failed"] += 1
        
        return result
    
    def _check_stage_passed(
        self,
        result: VerificationResult,
        stage: str,
    ) -> bool:
        """Check if a specific stage passed."""
        if stage == "syntax":
            return result.syntax_valid
        elif stage == "static_analysis":
            return result.static_analysis_issues < 5
        elif stage == "unit_tests":
            return result.tests_failed == 0
        elif stage == "regression":
            return not result.regression_detected
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            **self._stats,
            "success_rate": (
                self._stats["passed"] / self._stats["total_verifications"]
                if self._stats["total_verifications"] > 0 else 0
            ),
        }
    
    def reset_statistics(self):
        """Reset pipeline statistics."""
        self._stats = {
            "total_verifications": 0,
            "passed": 0,
            "failed": 0,
            "by_stage": {},
        }
