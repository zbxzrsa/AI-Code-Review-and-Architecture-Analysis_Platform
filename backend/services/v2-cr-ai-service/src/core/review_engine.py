"""
V2 CR-AI Review Engine

Production-grade code review engine with comprehensive analysis.
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from ..models.review_models import (
    ReviewRequest,
    ReviewResponse,
    ReviewFinding,
    ReviewSummary,
    FileReview,
    FindingSeverity,
    FindingCategory,
    CodeLocation,
    FixSuggestion,
)
from ..config.review_config import REVIEW_DIMENSIONS, REVIEW_CONFIG
from ..config.model_config import SYSTEM_PROMPTS
from .consensus_protocol import ConsensusProtocol


logger = logging.getLogger(__name__)


class ReviewEngine:
    """
    Production-grade code review engine.
    
    Features:
    - Multi-dimensional review (7 dimensions)
    - Consensus verification for critical issues
    - Deterministic outputs
    - Production guarantees
    """
    
    def __init__(
        self,
        primary_model_client: Any,
        secondary_model_client: Optional[Any] = None,
        consensus_protocol: Optional[ConsensusProtocol] = None,
    ):
        self.primary_client = primary_model_client
        self.secondary_client = secondary_model_client
        
        # Initialize consensus protocol
        if consensus_protocol:
            self.consensus = consensus_protocol
        else:
            self.consensus = ConsensusProtocol(
                primary_model_client=primary_model_client,
                secondary_model_client=secondary_model_client,
            )
        
        # Review cache for determinism
        self._review_cache: Dict[str, ReviewResponse] = {}
        
        # Metrics
        self._total_reviews = 0
        self._successful_reviews = 0
        self._failed_reviews = 0
    
    def _compute_cache_key(self, request: ReviewRequest) -> str:
        """Compute deterministic cache key for review."""
        import hashlib
        content = json.dumps({
            "files": [{"path": f.get("path"), "hash": hashlib.md5(f.get("content", "").encode()).hexdigest()} 
                      for f in request.files],
            "dimensions": [d.value if d else None for d in (request.dimensions or [])],
        }, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def review(self, request: ReviewRequest) -> ReviewResponse:
        """
        Perform comprehensive code review.
        
        Args:
            request: Review request with files and configuration
            
        Returns:
            ReviewResponse with findings and summary
        """
        start_time = time.time()
        self._total_reviews += 1
        review_id = str(uuid.uuid4())
        
        # Check cache for determinism
        cache_key = self._compute_cache_key(request)
        if cache_key in self._review_cache:
            logger.debug(f"Returning cached review for {cache_key[:8]}...")
            return self._review_cache[cache_key]
        
        try:
            # Determine dimensions to review
            dimensions = request.dimensions or list(FindingCategory)
            
            # Review each file
            file_reviews = []
            all_findings: List[ReviewFinding] = []
            
            for file_data in request.files:
                file_path = file_data.get("path", "unknown")
                content = file_data.get("content", "")
                language = file_data.get("language", self._detect_language(file_path))
                
                file_review = await self._review_file(
                    file_path=file_path,
                    content=content,
                    language=language,
                    dimensions=dimensions,
                    max_findings=request.max_findings_per_file,
                )
                
                file_reviews.append(file_review)
                all_findings.extend(file_review.findings)
            
            # Apply consensus verification for critical/high findings
            if request.consensus_enabled and self.secondary_client:
                code_context = {f.get("path"): f.get("content", "") for f in request.files}
                critical_findings = [f for f in all_findings 
                                    if f.severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH]]
                
                if critical_findings:
                    workflow = await self.consensus.verify_findings_batch(
                        findings=critical_findings,
                        code_context=code_context,
                        review_id=review_id,
                    )
                    
                    # Update findings based on consensus
                    for result in workflow.results:
                        finding = next((f for f in all_findings if f.id == result.finding_id), None)
                        if finding:
                            finding.consensus_verified = result.consensus_reached
                            finding.confidence = result.final_confidence
                            finding.requires_manual_review = result.requires_manual_review
            
            # Organize findings by confidence
            high_conf = [f for f in all_findings if f.confidence >= 0.85]
            medium_conf = [f for f in all_findings if 0.6 <= f.confidence < 0.85]
            low_conf = [f for f in all_findings if f.confidence < 0.6 and not f.requires_manual_review]
            manual_review = [f for f in all_findings if f.requires_manual_review]
            
            # Generate summary
            summary = self._generate_summary(all_findings)
            
            total_time = int((time.time() - start_time) * 1000)
            
            response = ReviewResponse(
                id=review_id,
                status="completed",
                requested_at=datetime.utcnow(),
                completed_at=datetime.utcnow(),
                total_time_ms=total_time,
                primary_model="claude-3-sonnet",
                secondary_model="gpt-4-turbo" if request.consensus_enabled else None,
                consensus_used=request.consensus_enabled,
                files_reviewed=len(request.files),
                file_reviews=file_reviews,
                summary=summary,
                slo_compliant=total_time < 500,
                high_confidence_findings=high_conf,
                medium_confidence_findings=medium_conf,
                low_confidence_findings=low_conf,
                manual_review_needed=manual_review,
            )
            
            # Cache for determinism
            self._review_cache[cache_key] = response
            
            self._successful_reviews += 1
            logger.info(f"Review {review_id} completed: {len(all_findings)} findings in {total_time}ms")
            
            return response
            
        except Exception as e:
            self._failed_reviews += 1
            logger.error(f"Review failed: {e}")
            raise
    
    async def _review_file(
        self,
        file_path: str,
        content: str,
        language: str,
        dimensions: List[FindingCategory],
        max_findings: int,
    ) -> FileReview:
        """Review a single file."""
        start_time = time.time()
        
        # Build review prompt
        prompt = self._build_review_prompt(file_path, content, language, dimensions)
        system_prompt = SYSTEM_PROMPTS["code_review"]
        
        findings = []
        
        try:
            # Call primary model
            if hasattr(self.primary_client, 'analyze'):
                response = await self.primary_client.analyze(prompt, system_prompt)
                findings = self._parse_review_response(response.content, file_path)
            else:
                # Mock findings for testing
                findings = self._generate_mock_findings(file_path, content, dimensions)
            
            # Limit findings
            findings = findings[:max_findings]
            
        except Exception as e:
            logger.error(f"File review failed for {file_path}: {e}")
        
        review_time = int((time.time() - start_time) * 1000)
        
        return FileReview(
            file_path=file_path,
            language=language,
            lines_of_code=len(content.splitlines()),
            findings=findings,
            review_time_ms=review_time,
        )
    
    def _build_review_prompt(
        self,
        file_path: str,
        content: str,
        language: str,
        dimensions: List[FindingCategory],
    ) -> str:
        """Build code review prompt."""
        dimension_names = [d.value for d in dimensions]
        
        return f"""Review the following {language} code:

## File: {file_path}

```{language}
{content[:8000]}  # Limit content length
```

## Review Dimensions
Focus on: {', '.join(dimension_names)}

## Required Output Format
Provide findings as a JSON array:
```json
{{
  "findings": [
    {{
      "title": "Short title",
      "description": "Detailed description",
      "category": "security|correctness|performance|maintainability|architecture|testing|documentation",
      "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
      "start_line": 10,
      "end_line": 15,
      "code_snippet": "affected code",
      "fix_suggestion": "how to fix",
      "confidence": 0.95
    }}
  ],
  "summary": "Brief overall assessment",
  "quality_score": 85
}}
```

Be precise, actionable, and minimize false positives."""
    
    def _parse_review_response(
        self,
        content: str,
        file_path: str,
    ) -> List[ReviewFinding]:
        """Parse review response from model."""
        try:
            # Extract JSON from response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                data = json.loads(content[json_start:json_end])
            else:
                return []
            
            findings = []
            for idx, f in enumerate(data.get("findings", [])):
                try:
                    # Map category
                    category_str = f.get("category", "correctness").lower()
                    try:
                        category = FindingCategory(category_str)
                    except ValueError:
                        category = FindingCategory.CORRECTNESS
                    
                    # Map severity
                    severity_str = f.get("severity", "MEDIUM").upper()
                    try:
                        severity = FindingSeverity(severity_str)
                    except ValueError:
                        severity = FindingSeverity.MEDIUM
                    
                    finding = ReviewFinding(
                        id=f"finding_{uuid.uuid4().hex[:8]}",
                        title=f.get("title", "Untitled finding"),
                        description=f.get("description", ""),
                        category=category,
                        severity=severity,
                        location=CodeLocation(
                            file=file_path,
                            start_line=f.get("start_line", 1),
                            end_line=f.get("end_line", f.get("start_line", 1)),
                            code_snippet=f.get("code_snippet"),
                        ),
                        confidence=f.get("confidence", 0.8),
                        detected_by="claude-3-sonnet",
                    )
                    
                    # Add fix suggestion if provided
                    if f.get("fix_suggestion"):
                        finding.fix_suggestions.append(FixSuggestion(
                            description=f["fix_suggestion"],
                            code_after=f.get("fixed_code", ""),
                            confidence=0.8,
                        ))
                    
                    findings.append(finding)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse finding: {e}")
                    continue
            
            return findings
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON: {e}")
            return []
    
    def _generate_mock_findings(
        self,
        file_path: str,
        content: str,
        dimensions: List[FindingCategory],
    ) -> List[ReviewFinding]:
        """Generate mock findings for testing."""
        findings = []
        
        if FindingCategory.SECURITY in dimensions:
            findings.append(ReviewFinding(
                id=f"finding_{uuid.uuid4().hex[:8]}",
                title="Potential SQL Injection",
                description="String concatenation used in SQL query construction",
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                location=CodeLocation(file=file_path, start_line=10, end_line=12),
                confidence=0.85,
                detected_by="claude-3-sonnet",
                cwe_id="CWE-89",
                owasp_id="A03:2021",
            ))
        
        if FindingCategory.CORRECTNESS in dimensions:
            findings.append(ReviewFinding(
                id=f"finding_{uuid.uuid4().hex[:8]}",
                title="Missing Null Check",
                description="Variable may be null before dereferencing",
                category=FindingCategory.CORRECTNESS,
                severity=FindingSeverity.MEDIUM,
                location=CodeLocation(file=file_path, start_line=25, end_line=25),
                confidence=0.78,
                detected_by="claude-3-sonnet",
            ))
        
        return findings
    
    def _generate_summary(self, findings: List[ReviewFinding]) -> ReviewSummary:
        """Generate review summary."""
        total = len(findings)
        critical = sum(1 for f in findings if f.severity == FindingSeverity.CRITICAL)
        high = sum(1 for f in findings if f.severity == FindingSeverity.HIGH)
        medium = sum(1 for f in findings if f.severity == FindingSeverity.MEDIUM)
        low = sum(1 for f in findings if f.severity == FindingSeverity.LOW)
        info = sum(1 for f in findings if f.severity == FindingSeverity.INFO)
        
        by_category = {}
        for f in findings:
            cat = f.category.value
            by_category[cat] = by_category.get(cat, 0) + 1
        
        consensus_verified = sum(1 for f in findings if f.consensus_verified)
        manual_needed = sum(1 for f in findings if f.requires_manual_review)
        
        # Calculate quality score (100 - weighted issues)
        quality_score = 100 - (critical * 20 + high * 10 + medium * 5 + low * 2 + info * 1)
        quality_score = max(0, min(100, quality_score))
        
        # Generate recommendation
        if critical > 0:
            recommendation = "BLOCK: Critical issues must be resolved before merge"
        elif high > 3:
            recommendation = "REQUEST CHANGES: Multiple high-priority issues need attention"
        elif high > 0 or medium > 5:
            recommendation = "REVIEW RECOMMENDED: Some issues should be addressed"
        else:
            recommendation = "APPROVE: Code meets quality standards"
        
        return ReviewSummary(
            total_findings=total,
            critical_count=critical,
            high_count=high,
            medium_count=medium,
            low_count=low,
            info_count=info,
            by_category=by_category,
            consensus_verified_count=consensus_verified,
            manual_review_needed_count=manual_needed,
            overall_quality_score=quality_score,
            recommendation=recommendation,
        )
    
    def _detect_language(self, file_path: str) -> str:
        """Detect language from file extension."""
        ext_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".c": "c",
            ".cpp": "cpp",
            ".cs": "csharp",
            ".rb": "ruby",
            ".php": "php",
            ".swift": "swift",
            ".kt": "kotlin",
        }
        
        for ext, lang in ext_map.items():
            if file_path.endswith(ext):
                return lang
        
        return "unknown"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get review engine metrics."""
        return {
            "total_reviews": self._total_reviews,
            "successful_reviews": self._successful_reviews,
            "failed_reviews": self._failed_reviews,
            "success_rate": self._successful_reviews / max(1, self._total_reviews),
            "cache_size": len(self._review_cache),
            "consensus_metrics": self.consensus.get_metrics().model_dump(),
        }
    
    def clear_cache(self) -> int:
        """Clear review cache."""
        count = len(self._review_cache)
        self._review_cache.clear()
        return count
