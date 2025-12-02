"""
Impact Predictor for V1 VC-AI

Predicts the impact of code changes using:
- Dependency graph analysis
- AST-based symbol extraction
- Blast radius estimation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional, Tuple
from enum import Enum
import re


class DependencyType(str, Enum):
    """Types of code dependencies"""
    IMPORT = "import"
    FUNCTION_CALL = "function_call"
    CLASS_INHERITANCE = "class_inheritance"
    INTERFACE_IMPLEMENTATION = "interface_implementation"
    TYPE_REFERENCE = "type_reference"
    FILE_INCLUDE = "file_include"


@dataclass
class Symbol:
    """A code symbol (function, class, variable, etc.)"""
    name: str
    symbol_type: str  # function, class, variable, constant
    file_path: str
    line_number: int
    scope: str = ""  # module.class.method


@dataclass
class Dependency:
    """A dependency between symbols"""
    source: Symbol
    target: Symbol
    dependency_type: DependencyType
    weight: float = 1.0


@dataclass
class DependencyGraph:
    """Graph of code dependencies"""
    symbols: Dict[str, Symbol] = field(default_factory=dict)
    edges: List[Dependency] = field(default_factory=list)
    
    def add_symbol(self, symbol: Symbol):
        key = f"{symbol.file_path}:{symbol.name}"
        self.symbols[key] = symbol
    
    def add_dependency(self, dep: Dependency):
        self.edges.append(dep)
    
    def get_dependents(self, symbol_key: str) -> List[str]:
        """Get all symbols that depend on the given symbol"""
        return [
            f"{e.source.file_path}:{e.source.name}"
            for e in self.edges
            if f"{e.target.file_path}:{e.target.name}" == symbol_key
        ]
    
    def get_dependencies(self, symbol_key: str) -> List[str]:
        """Get all symbols that the given symbol depends on"""
        return [
            f"{e.target.file_path}:{e.target.name}"
            for e in self.edges
            if f"{e.source.file_path}:{e.source.name}" == symbol_key
        ]


@dataclass
class ImpactPrediction:
    """Predicted impact of a change"""
    changed_symbols: List[Symbol]
    directly_affected: List[Symbol]
    indirectly_affected: List[Symbol]
    blast_radius: int
    risk_level: str  # low, medium, high, critical
    confidence: float
    affected_modules: List[str]
    affected_tests: List[str]
    breaking_changes: List[str]


class ImpactPredictor:
    """
    Predicts impact of code changes.
    
    Uses dependency analysis and heuristics to estimate:
    - Direct impact (immediate dependents)
    - Indirect impact (transitive dependents)
    - Blast radius (total affected code)
    - Risk level
    """
    
    # Patterns for extracting dependencies
    IMPORT_PATTERNS = {
        "python": [
            r'from\s+([\w.]+)\s+import',
            r'import\s+([\w.]+)',
        ],
        "javascript": [
            r'import\s+.*\s+from\s+[\'"](.+?)[\'"]',
            r'require\([\'"](.+?)[\'"]\)',
        ],
        "typescript": [
            r'import\s+.*\s+from\s+[\'"](.+?)[\'"]',
        ],
        "java": [
            r'import\s+([\w.]+);',
        ],
        "go": [
            r'import\s+"(.+?)"',
            r'import\s+\(\s*"(.+?)"',
        ],
    }
    
    def __init__(self):
        self.dependency_graph = DependencyGraph()
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns"""
        self.compiled_patterns = {
            lang: [re.compile(p) for p in patterns]
            for lang, patterns in self.IMPORT_PATTERNS.items()
        }
    
    def build_dependency_graph(
        self,
        files: Dict[str, str],
        language: str = "python",
    ) -> DependencyGraph:
        """
        Build dependency graph from source files.
        
        Args:
            files: Dict of file_path -> content
            language: Programming language
            
        Returns:
            DependencyGraph with extracted dependencies
        """
        graph = DependencyGraph()
        
        for file_path, content in files.items():
            # Extract symbols
            symbols = self._extract_symbols(file_path, content, language)
            for symbol in symbols:
                graph.add_symbol(symbol)
            
            # Extract dependencies
            dependencies = self._extract_dependencies(file_path, content, language)
            for dep in dependencies:
                graph.add_dependency(dep)
        
        self.dependency_graph = graph
        return graph
    
    def _extract_symbols(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Symbol]:
        """Extract symbols from source code"""
        symbols = []
        lines = content.split('\n')
        
        # Simple pattern-based extraction
        patterns = {
            "python": {
                "function": r'def\s+(\w+)\s*\(',
                "class": r'class\s+(\w+)',
                "variable": r'^(\w+)\s*=',
            },
            "javascript": {
                "function": r'function\s+(\w+)\s*\(|(\w+)\s*=\s*(?:async\s*)?\(',
                "class": r'class\s+(\w+)',
                "variable": r'(?:const|let|var)\s+(\w+)',
            },
        }
        
        lang_patterns = patterns.get(language, patterns["python"])
        
        for i, line in enumerate(lines, 1):
            for symbol_type, pattern in lang_patterns.items():
                match = re.search(pattern, line)
                if match:
                    name = match.group(1) or (match.group(2) if len(match.groups()) > 1 else None)
                    if name:
                        symbols.append(Symbol(
                            name=name,
                            symbol_type=symbol_type,
                            file_path=file_path,
                            line_number=i,
                        ))
        
        return symbols
    
    def _extract_dependencies(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[Dependency]:
        """Extract dependencies from source code"""
        dependencies = []
        
        patterns = self.compiled_patterns.get(language, self.compiled_patterns.get("python", []))
        
        # File as source symbol
        file_symbol = Symbol(
            name=file_path.split('/')[-1],
            symbol_type="file",
            file_path=file_path,
            line_number=0,
        )
        
        for pattern in patterns:
            for match in pattern.finditer(content):
                target_name = match.group(1)
                target_symbol = Symbol(
                    name=target_name,
                    symbol_type="module",
                    file_path=target_name,
                    line_number=0,
                )
                
                dependencies.append(Dependency(
                    source=file_symbol,
                    target=target_symbol,
                    dependency_type=DependencyType.IMPORT,
                ))
        
        return dependencies
    
    async def predict_impact(
        self,
        changed_files: List[str],
        diff_content: str,
        graph: Optional[DependencyGraph] = None,
    ) -> ImpactPrediction:
        """
        Predict the impact of code changes.
        
        Args:
            changed_files: List of changed file paths
            diff_content: Git diff content
            graph: Dependency graph (uses internal if not provided)
            
        Returns:
            ImpactPrediction with estimated impact
        """
        graph = graph or self.dependency_graph
        
        # Find changed symbols
        changed_symbols = []
        for file_path in changed_files:
            symbols = [s for s in graph.symbols.values() if s.file_path == file_path]
            changed_symbols.extend(symbols)
        
        # Find directly affected symbols
        directly_affected = set()
        for symbol in changed_symbols:
            key = f"{symbol.file_path}:{symbol.name}"
            dependents = graph.get_dependents(key)
            for dep_key in dependents:
                if dep_key in graph.symbols:
                    directly_affected.add(graph.symbols[dep_key])
        
        # Find indirectly affected (transitive closure)
        indirectly_affected = set()
        to_process = list(directly_affected)
        processed = set()
        
        while to_process:
            symbol = to_process.pop()
            key = f"{symbol.file_path}:{symbol.name}"
            
            if key in processed:
                continue
            processed.add(key)
            
            dependents = graph.get_dependents(key)
            for dep_key in dependents:
                if dep_key in graph.symbols and dep_key not in processed:
                    dep_symbol = graph.symbols[dep_key]
                    indirectly_affected.add(dep_symbol)
                    to_process.append(dep_symbol)
        
        # Remove directly affected from indirectly affected
        indirectly_affected -= directly_affected
        
        # Calculate blast radius
        blast_radius = len(changed_symbols) + len(directly_affected) + len(indirectly_affected)
        
        # Determine risk level
        if blast_radius > 50:
            risk_level = "critical"
        elif blast_radius > 20:
            risk_level = "high"
        elif blast_radius > 5:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Find affected modules
        affected_modules = set()
        for symbol in list(directly_affected) + list(indirectly_affected):
            parts = symbol.file_path.split('/')
            if len(parts) > 1:
                affected_modules.add(parts[0])
        
        # Find affected tests
        affected_tests = [
            s.file_path for s in list(directly_affected) + list(indirectly_affected)
            if 'test' in s.file_path.lower()
        ]
        
        # Detect breaking changes
        breaking_changes = self._detect_breaking_changes(diff_content)
        
        return ImpactPrediction(
            changed_symbols=changed_symbols,
            directly_affected=list(directly_affected),
            indirectly_affected=list(indirectly_affected),
            blast_radius=blast_radius,
            risk_level=risk_level,
            confidence=0.8,
            affected_modules=list(affected_modules),
            affected_tests=affected_tests,
            breaking_changes=breaking_changes,
        )
    
    def _detect_breaking_changes(self, diff: str) -> List[str]:
        """Detect potential breaking changes in diff"""
        breaking_changes = []
        
        patterns = [
            (r'-\s*def\s+(\w+)', "Removed function: {}"),
            (r'-\s*class\s+(\w+)', "Removed class: {}"),
            (r'-\s*public\s+', "Removed public method/field"),
            (r'def\s+(\w+).*:\s*$.*raise\s+NotImplementedError', "Deprecated function: {}"),
        ]
        
        for pattern, message_template in patterns:
            for match in re.finditer(pattern, diff, re.MULTILINE):
                name = match.group(1) if match.lastindex else ""
                breaking_changes.append(message_template.format(name))
        
        return breaking_changes[:5]  # Limit to top 5
