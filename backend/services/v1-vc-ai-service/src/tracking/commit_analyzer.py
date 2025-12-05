"""
Commit Analysis Engine for V1 VC-AI

Provides semantic understanding of commits:
- Change type classification
- Impact level prediction
- Affected module extraction
- Semantic embedding generation
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
from datetime import datetime
import re
import hashlib
import logging

import torch
import torch.nn.functional as F


logger = logging.getLogger(__name__)


class ChangeType(str, Enum):
    """Types of code changes"""
    BUG_FIX = "bug_fix"
    FEATURE = "feature"
    REFACTOR = "refactor"
    OPTIMIZATION = "optimization"
    DOCS = "docs"
    TEST = "test"
    CHORE = "chore"
    SECURITY = "security"
    BREAKING = "breaking"
    UNKNOWN = "unknown"


class ImpactLevel(str, Enum):
    """Impact severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class FileChange:
    """Represents changes to a single file"""
    path: str
    additions: int
    deletions: int
    change_type: str  # added, modified, deleted, renamed
    language: Optional[str] = None
    functions_changed: List[str] = field(default_factory=list)
    classes_changed: List[str] = field(default_factory=list)


@dataclass
class CommitAnalysisResult:
    """Complete analysis result for a commit"""
    commit_hash: str
    message: str
    timestamp: datetime
    
    # Classification results
    change_type: ChangeType
    change_type_confidence: float
    impact_level: ImpactLevel
    impact_confidence: float
    
    # Affected components
    files_changed: List[FileChange]
    modules_affected: List[str]
    functions_affected: List[str]
    classes_affected: List[str]
    
    # Semantic analysis
    semantic_embedding: Optional[torch.Tensor] = None
    explanation: str = ""
    key_changes: List[str] = field(default_factory=list)
    
    # Risk assessment
    risk_score: float = 0.0
    risk_factors: List[str] = field(default_factory=list)
    
    # Metrics
    total_additions: int = 0
    total_deletions: int = 0
    complexity_delta: float = 0.0
    
    # Timing
    analysis_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "commit_hash": self.commit_hash,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "change_type": self.change_type.value,
            "change_type_confidence": self.change_type_confidence,
            "impact_level": self.impact_level.value,
            "impact_confidence": self.impact_confidence,
            "modules_affected": self.modules_affected,
            "functions_affected": self.functions_affected,
            "classes_affected": self.classes_affected,
            "explanation": self.explanation,
            "key_changes": self.key_changes,
            "risk_score": self.risk_score,
            "risk_factors": self.risk_factors,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "analysis_time_ms": self.analysis_time_ms,
        }


class CommitAnalyzer:
    """
    Main commit analysis engine.
    
    Responsibilities:
    1. Tokenize commit message and diff
    2. Generate semantic embeddings
    3. Classify change type
    4. Predict impact level
    5. Extract affected components
    """
    
    # Patterns for detecting change types from commit messages
    CHANGE_TYPE_PATTERNS = {
        ChangeType.BUG_FIX: [
            r'\bfix(es|ed)?\b', r'\bbug\b', r'\bpatch\b', r'\bresolve[sd]?\b',
            r'\bhotfix\b', r'\bregression\b', r'\bcorrect\b',
        ],
        ChangeType.FEATURE: [
            r'\badd(s|ed)?\b', r'\bfeat(ure)?\b', r'\bimplement\b', 
            r'\bintroduce\b', r'\bnew\b', r'\benable\b',
        ],
        ChangeType.REFACTOR: [
            r'\brefactor\b', r'\brestructure\b', r'\breorganize\b',
            r'\bclean\s?up\b', r'\bsimplify\b', r'\bextract\b',
        ],
        ChangeType.OPTIMIZATION: [
            r'\boptimize\b', r'\bperformance\b', r'\bspeed\b', r'\bfast(er)?\b',
            r'\bcache\b', r'\befficient\b', r'\breduce\b',
        ],
        ChangeType.DOCS: [
            r'\bdoc(s|umentation)?\b', r'\breadme\b', r'\bcomment\b',
            r'\btypedoc\b', r'\bjsdoc\b', r'\bchangelog\b',
        ],
        ChangeType.TEST: [
            r'\btest(s|ing)?\b', r'\bspec\b', r'\bcoverage\b',
            r'\bmock\b', r'\bfixture\b', r'\be2e\b',
        ],
        ChangeType.CHORE: [
            r'\bchore\b', r'\bbump\b', r'\bupdate\b', r'\bupgrade\b',
            r'\bdeps?\b', r'\bdependenc(y|ies)\b', r'\bci\b', r'\bcd\b',
        ],
        ChangeType.SECURITY: [
            r'\bsecurity\b', r'\bvulnerabilit(y|ies)\b', r'\bcve\b',
            r'\bauth\b', r'\bpermission\b', r'\bencrypt\b',
        ],
        ChangeType.BREAKING: [
            r'\bbreaking\b', r'\b!:\b', r'\bBREAKING CHANGE\b',
            r'\bincompatible\b', r'\bremove[sd]?\b',
        ],
    }
    
    # Patterns for extracting code entities
    FUNCTION_PATTERNS = [
        r'def\s+(\w+)\s*\(',                    # Python
        r'function\s+(\w+)\s*\(',               # JavaScript
        r'(async\s+)?(\w+)\s*=\s*(?:async\s*)?\(',  # Arrow functions
        r'func\s+(\w+)\s*\(',                   # Go
        r'fn\s+(\w+)\s*\(',                     # Rust
        r'public\s+\w+\s+(\w+)\s*\(',           # Java/C#
    ]
    
    CLASS_PATTERNS = [
        r'class\s+(\w+)',                       # Python/JavaScript/Java
        r'struct\s+(\w+)',                      # Go/Rust/C
        r'interface\s+(\w+)',                   # TypeScript/Java
        r'type\s+(\w+)\s+struct',               # Go
    ]
    
    def __init__(
        self,
        model: Optional[Any] = None,
        tokenizer: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        
        # Compile regex patterns
        self._compile_patterns()
        
        # Metrics tracking
        self.metrics = {
            "total_analyzed": 0,
            "avg_analysis_time_ms": 0.0,
            "change_type_distribution": {ct.value: 0 for ct in ChangeType},
            "impact_distribution": {il.value: 0 for il in ImpactLevel},
        }
    
    def _compile_patterns(self):
        """Compile all regex patterns for efficiency"""
        self.compiled_change_patterns = {
            ct: [re.compile(p, re.IGNORECASE) for p in patterns]
            for ct, patterns in self.CHANGE_TYPE_PATTERNS.items()
        }
        
        self.compiled_function_patterns = [
            re.compile(p) for p in self.FUNCTION_PATTERNS
        ]
        
        self.compiled_class_patterns = [
            re.compile(p) for p in self.CLASS_PATTERNS
        ]
    
    async def analyze_commit(
        self,
        commit_hash: str,
        message: str,
        diff: str,
        timestamp: Optional[datetime] = None,
        use_model: bool = True,
    ) -> CommitAnalysisResult:
        """
        Analyze a single commit.
        
        Args:
            commit_hash: Git commit hash
            message: Commit message
            diff: Git diff content
            timestamp: Commit timestamp
            use_model: Whether to use the ML model for analysis
            
        Returns:
            Complete analysis result
        """
        import time
        start_time = time.time()
        
        timestamp = timestamp or datetime.now(timezone.utc)
        
        # Parse diff to extract file changes
        file_changes = self._parse_diff(diff)
        
        # Extract affected components
        modules, functions, classes = self._extract_components(diff, file_changes)
        
        # Classify change type
        if use_model and self.model is not None:
            change_type, change_confidence, embedding = await self._model_classify(
                message, diff
            )
        else:
            change_type, change_confidence = self._rule_based_classify(message)
            embedding = None
        
        # Predict impact level
        if use_model and self.model is not None:
            impact_level, impact_confidence = await self._model_predict_impact(
                message, diff, file_changes
            )
        else:
            impact_level, impact_confidence = self._rule_based_impact(
                file_changes, modules, functions
            )
        
        # Assess risk
        risk_score, risk_factors = self._assess_risk(
            change_type, impact_level, file_changes, modules
        )
        
        # Generate explanation
        explanation = self._generate_explanation(
            change_type, impact_level, file_changes, modules
        )
        
        # Extract key changes
        key_changes = self._extract_key_changes(diff, file_changes)
        
        # Calculate totals
        total_additions = sum(fc.additions for fc in file_changes)
        total_deletions = sum(fc.deletions for fc in file_changes)
        
        analysis_time_ms = (time.time() - start_time) * 1000
        
        # Update metrics
        self._update_metrics(change_type, impact_level, analysis_time_ms)
        
        return CommitAnalysisResult(
            commit_hash=commit_hash,
            message=message,
            timestamp=timestamp,
            change_type=change_type,
            change_type_confidence=change_confidence,
            impact_level=impact_level,
            impact_confidence=impact_confidence,
            files_changed=file_changes,
            modules_affected=modules,
            functions_affected=functions,
            classes_affected=classes,
            semantic_embedding=embedding,
            explanation=explanation,
            key_changes=key_changes,
            risk_score=risk_score,
            risk_factors=risk_factors,
            total_additions=total_additions,
            total_deletions=total_deletions,
            analysis_time_ms=analysis_time_ms,
        )
    
    def _parse_diff(self, diff: str) -> List[FileChange]:
        """Parse git diff to extract file changes"""
        file_changes = []
        current_file = None
        additions = 0
        deletions = 0
        
        for line in diff.split('\n'):
            # New file header
            if line.startswith('diff --git'):
                # Save previous file
                if current_file:
                    file_changes.append(FileChange(
                        path=current_file,
                        additions=additions,
                        deletions=deletions,
                        change_type="modified",
                        language=self._detect_language(current_file),
                    ))
                
                # Extract new file path
                match = re.search(r'a/(.+?) b/', line)
                current_file = match.group(1) if match else "unknown"
                additions = 0
                deletions = 0
            
            # Count additions/deletions
            elif line.startswith('+') and not line.startswith('+++'):
                additions += 1
            elif line.startswith('-') and not line.startswith('---'):
                deletions += 1
        
        # Don't forget the last file
        if current_file:
            file_changes.append(FileChange(
                path=current_file,
                additions=additions,
                deletions=deletions,
                change_type="modified",
                language=self._detect_language(current_file),
            ))
        
        return file_changes
    
    def _detect_language(self, file_path: str) -> Optional[str]:
        """Detect programming language from file extension"""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.tsx': 'typescript',
            '.jsx': 'javascript',
            '.java': 'java',
            '.go': 'go',
            '.rs': 'rust',
            '.rb': 'ruby',
            '.php': 'php',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.cs': 'csharp',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.md': 'markdown',
            '.yaml': 'yaml',
            '.yml': 'yaml',
            '.json': 'json',
            '.xml': 'xml',
            '.sql': 'sql',
            '.sh': 'shell',
            '.bash': 'shell',
        }
        
        for ext, lang in extension_map.items():
            if file_path.endswith(ext):
                return lang
        return None
    
    def _extract_components(
        self,
        diff: str,
        file_changes: List[FileChange],
    ) -> Tuple[List[str], List[str], List[str]]:
        """Extract affected modules, functions, and classes"""
        modules = set()
        functions = set()
        classes = set()
        
        # Extract modules from file paths
        for fc in file_changes:
            parts = fc.path.split('/')
            if len(parts) > 1:
                modules.add(parts[0])
                if len(parts) > 2:
                    modules.add('/'.join(parts[:2]))
        
        # Extract functions from diff
        for pattern in self.compiled_function_patterns:
            for match in pattern.finditer(diff):
                func_name = match.group(1) if match.group(1) else match.group(2)
                if func_name:
                    functions.add(func_name)
        
        # Extract classes from diff
        for pattern in self.compiled_class_patterns:
            for match in pattern.finditer(diff):
                classes.add(match.group(1))
        
        return list(modules), list(functions), list(classes)
    
    def _rule_based_classify(
        self,
        message: str,
    ) -> Tuple[ChangeType, float]:
        """Classify change type using rule-based patterns"""
        scores = {}
        
        for change_type, patterns in self.compiled_change_patterns.items():
            score = sum(1 for p in patterns if p.search(message))
            scores[change_type] = score
        
        if not any(scores.values()):
            return ChangeType.UNKNOWN, 0.5
        
        best_type = max(scores, key=scores.get)
        max_score = scores[best_type]
        total_score = sum(scores.values())
        confidence = max_score / total_score if total_score > 0 else 0.5
        
        return best_type, min(confidence + 0.3, 0.95)
    
    def _rule_based_impact(
        self,
        file_changes: List[FileChange],
        modules: List[str],
        functions: List[str],  # noqa: ARG002 - Reserved for function-level analysis
    ) -> Tuple[ImpactLevel, float]:
        """Predict impact level using rules"""
        total_changes = sum(fc.additions + fc.deletions for fc in file_changes)
        num_files = len(file_changes)
        num_modules = len(modules)
        
        # Heuristic scoring
        score = 0
        
        # Size-based scoring
        if total_changes > 500:
            score += 3
        elif total_changes > 100:
            score += 2
        elif total_changes > 20:
            score += 1
        
        # File count scoring
        if num_files > 10:
            score += 2
        elif num_files > 5:
            score += 1
        
        # Module breadth scoring
        if num_modules > 3:
            score += 2
        elif num_modules > 1:
            score += 1
        
        # Critical file detection
        critical_patterns = ['security', 'auth', 'database', 'migration', 'config']
        for fc in file_changes:
            if any(p in fc.path.lower() for p in critical_patterns):
                score += 2
                break
        
        # Map score to impact level
        if score >= 6:
            return ImpactLevel.CRITICAL, 0.85
        elif score >= 4:
            return ImpactLevel.HIGH, 0.80
        elif score >= 2:
            return ImpactLevel.MEDIUM, 0.75
        else:
            return ImpactLevel.LOW, 0.70
    
    def _model_classify(
        self,
        message: str,
        diff: str,
    ) -> Tuple[ChangeType, float, torch.Tensor]:
        """Classify using the ML model"""
        if self.model is None or self.tokenizer is None:
            change_type, confidence = self._rule_based_classify(message)
            return change_type, confidence, None
        
        # Tokenize input
        combined_input = f"{message}\n\n{diff[:2000]}"  # Truncate diff
        inputs = self.tokenizer.encode(combined_input, max_length=2048)
        input_ids = inputs["input_ids"].to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        
        # Get change type prediction
        change_logits = outputs.change_type_logits
        probs = F.softmax(change_logits, dim=-1)
        predicted_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, predicted_idx].item()
        
        change_type = list(ChangeType)[predicted_idx]
        embedding = outputs.embeddings
        
        return change_type, confidence, embedding
    
    def _model_predict_impact(
        self,
        message: str,
        diff: str,
        file_changes: List[FileChange],
    ) -> Tuple[ImpactLevel, float]:
        """Predict impact using the ML model"""
        if self.model is None or self.tokenizer is None:
            return self._rule_based_impact(file_changes, [], [])
        
        # Similar to classification
        combined_input = f"{message}\n\n{diff[:2000]}"
        inputs = self.tokenizer.encode(combined_input, max_length=2048)
        input_ids = inputs["input_ids"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)
        
        impact_logits = outputs.impact_logits
        probs = F.softmax(impact_logits, dim=-1)
        predicted_idx = probs.argmax(dim=-1).item()
        confidence = probs[0, predicted_idx].item()
        
        impact_level = list(ImpactLevel)[predicted_idx]
        
        return impact_level, confidence
    
    def _assess_risk(
        self,
        change_type: ChangeType,
        impact_level: ImpactLevel,
        file_changes: List[FileChange],
        modules: List[str],
    ) -> Tuple[float, List[str]]:
        """Assess risk score and factors"""
        risk_factors = []
        risk_score = 0.0
        
        # Impact-based risk
        impact_weights = {
            ImpactLevel.CRITICAL: 0.4,
            ImpactLevel.HIGH: 0.3,
            ImpactLevel.MEDIUM: 0.2,
            ImpactLevel.LOW: 0.1,
        }
        risk_score += impact_weights.get(impact_level, 0.2)
        
        # Change type risk
        if change_type == ChangeType.BREAKING:
            risk_score += 0.3
            risk_factors.append("Breaking change detected")
        elif change_type == ChangeType.SECURITY:
            risk_score += 0.25
            risk_factors.append("Security-related change")
        
        # Large change risk
        total_changes = sum(fc.additions + fc.deletions for fc in file_changes)
        if total_changes > 500:
            risk_score += 0.15
            risk_factors.append(f"Large change ({total_changes} lines)")
        
        # Multi-module risk
        if len(modules) > 3:
            risk_score += 0.1
            risk_factors.append(f"Affects multiple modules ({len(modules)})")
        
        # Critical file detection
        for fc in file_changes:
            if 'database' in fc.path.lower() or 'migration' in fc.path.lower():
                risk_score += 0.15
                risk_factors.append("Database/migration changes")
                break
        
        return min(risk_score, 1.0), risk_factors
    
    def _generate_explanation(
        self,
        change_type: ChangeType,
        impact_level: ImpactLevel,
        file_changes: List[FileChange],
        modules: List[str],
    ) -> str:
        """Generate human-readable explanation"""
        num_files = len(file_changes)
        total_additions = sum(fc.additions for fc in file_changes)
        total_deletions = sum(fc.deletions for fc in file_changes)
        
        explanation = f"This is a {change_type.value.replace('_', ' ')} commit "
        explanation += f"with {impact_level.value} impact. "
        explanation += f"It modifies {num_files} file(s) "
        explanation += f"(+{total_additions}/-{total_deletions} lines)"
        
        if modules:
            explanation += f" affecting {', '.join(modules[:3])}"
            if len(modules) > 3:
                explanation += f" and {len(modules) - 3} more modules"
        
        explanation += "."
        
        return explanation
    
    def _extract_key_changes(
        self,
        diff: str,
        file_changes: List[FileChange],
    ) -> List[str]:
        """Extract key changes from the diff"""
        key_changes = []
        
        # Look for significant patterns
        patterns = [
            (r'class\s+\w+', "New class defined"),
            (r'def\s+\w+', "New function defined"),
            (r'import\s+\w+', "New import added"),
            (r'TODO|FIXME|HACK', "Technical debt marker"),
            (r'raise\s+\w+Exception', "Exception handling change"),
            (r'@app\.route|@router', "API endpoint change"),
            (r'CREATE TABLE|ALTER TABLE', "Database schema change"),
        ]
        
        for pattern, description in patterns:
            if re.search(pattern, diff):
                key_changes.append(description)
        
        # Add file-level changes
        for fc in file_changes:
            if fc.additions > 100:
                key_changes.append(f"Major additions to {fc.path}")
            if fc.deletions > 100:
                key_changes.append(f"Major deletions from {fc.path}")
        
        return key_changes[:5]  # Limit to top 5
    
    def _update_metrics(
        self,
        change_type: ChangeType,
        impact_level: ImpactLevel,
        analysis_time_ms: float,
    ):
        """Update internal metrics"""
        self.metrics["total_analyzed"] += 1
        
        # Update running average
        n = self.metrics["total_analyzed"]
        old_avg = self.metrics["avg_analysis_time_ms"]
        self.metrics["avg_analysis_time_ms"] = old_avg + (analysis_time_ms - old_avg) / n
        
        self.metrics["change_type_distribution"][change_type.value] += 1
        self.metrics["impact_distribution"][impact_level.value] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.metrics.copy()
    
    def reset_metrics(self):
        """Reset metrics"""
        self.metrics = {
            "total_analyzed": 0,
            "avg_analysis_time_ms": 0.0,
            "change_type_distribution": {ct.value: 0 for ct in ChangeType},
            "impact_distribution": {il.value: 0 for il in ImpactLevel},
        }
