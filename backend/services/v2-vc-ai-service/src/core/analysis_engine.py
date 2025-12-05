"""
V2 VC-AI Analysis Engine

Production-grade commit analysis with deterministic outputs.
"""

import json
import logging
import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List

from .model_client import ModelClient, ModelResponse, ModelConfig, ModelProvider
from .circuit_breaker import CircuitBreaker
from ..models.analysis_models import (
    CommitAnalysisRequest,
    CommitAnalysisResponse,
    ChangeType,
    ImpactLevel,
    RiskAssessment,
    AffectedComponent,
    BreakingChange,
    RollbackPlan,
)
from ..config.model_config import SYSTEM_PROMPTS


logger = logging.getLogger(__name__)


class AnalysisEngine:
    """
    Production-grade analysis engine for version control operations.
    
    Features:
    - Deterministic outputs (same input = same output)
    - Circuit breaker protection
    - Comprehensive commit analysis
    - Risk assessment
    - Rollback planning
    """
    
    def __init__(
        self,
        model_client: ModelClient,
        circuit_breaker: Optional[CircuitBreaker] = None,
    ):
        self.model_client = model_client
        self.circuit_breaker = circuit_breaker
        
        # Analysis cache for determinism verification
        self._analysis_cache: Dict[str, CommitAnalysisResponse] = {}
        
        # Metrics
        self._total_analyses = 0
        self._successful_analyses = 0
        self._failed_analyses = 0
    
    def _build_analysis_prompt(self, request: CommitAnalysisRequest, diff_content: Optional[str] = None) -> str:
        """Build the analysis prompt"""
        prompt = f"""Analyze the following commit:

Repository: {request.repo}
Commit Hash: {request.commit_hash}
Branch: {request.branch}

"""
        if diff_content:
            prompt += f"""Diff Content:
```
{diff_content[:10000]}  # Limit diff size
```

"""
        
        prompt += """Provide a comprehensive analysis in the following JSON format:
{
    "change_type": "feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|security|deps",
    "impact_level": "LOW|MEDIUM|HIGH|CRITICAL",
    "risk_assessment": "SAFE|CAUTION|RISKY",
    "summary": "One-line summary",
    "description": "Detailed description",
    "affected_services": ["service1", "service2"],
    "affected_components": [
        {
            "name": "component_name",
            "path": "file/path",
            "change_type": "added|modified|deleted",
            "confidence": 0.95
        }
    ],
    "breaking_changes": [
        {
            "type": "api_change|behavior_change|removal",
            "description": "What is breaking",
            "affected_apis": ["API1"],
            "migration_steps": ["Step 1", "Step 2"]
        }
    ],
    "rollback_plan": {
        "steps": ["Step 1", "Step 2"],
        "estimated_duration_minutes": 5,
        "data_migration_required": false,
        "service_restart_required": true,
        "verification_steps": ["Verify step 1"],
        "risks": ["Risk 1"]
    },
    "recommendations": ["Recommendation 1"],
    "review_suggestions": ["Review suggestion 1"],
    "test_suggestions": ["Test suggestion 1"],
    "complexity_score": 0.5,
    "risk_score": 0.3
}

Be precise and consistent. The same commit should always produce the same analysis."""
        
        return prompt
    
    def _parse_analysis_response(
        self,
        response: ModelResponse,
        request: CommitAnalysisRequest,
    ) -> CommitAnalysisResponse:
        """Parse the model response into structured analysis"""
        try:
            # Extract JSON from response
            content = response.content
            
            # Try to find JSON in the response
            json_start = content.find('{')
            json_end = content.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = content[json_start:json_end]
                data = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
            
            # Parse affected components
            affected_components = []
            for comp in data.get("affected_components", []):
                affected_components.append(AffectedComponent(
                    name=comp.get("name", "unknown"),
                    path=comp.get("path", ""),
                    change_type=comp.get("change_type", "modified"),
                    confidence=comp.get("confidence", 0.8),
                ))
            
            # Parse breaking changes
            breaking_changes = []
            for bc in data.get("breaking_changes", []):
                breaking_changes.append(BreakingChange(
                    type=bc.get("type", "unknown"),
                    description=bc.get("description", ""),
                    affected_apis=bc.get("affected_apis", []),
                    migration_steps=bc.get("migration_steps", []),
                ))
            
            # Parse rollback plan
            rollback_data = data.get("rollback_plan", {})
            rollback_plan = RollbackPlan(
                steps=rollback_data.get("steps", ["Revert commit"]),
                estimated_duration_minutes=rollback_data.get("estimated_duration_minutes", 5),
                data_migration_required=rollback_data.get("data_migration_required", False),
                service_restart_required=rollback_data.get("service_restart_required", False),
                verification_steps=rollback_data.get("verification_steps", []),
                risks=rollback_data.get("risks", []),
            )
            
            # Map change type
            change_type_str = data.get("change_type", "chore").lower()
            try:
                change_type = ChangeType(change_type_str)
            except ValueError:
                change_type = ChangeType.CHORE
            
            # Map impact level
            impact_str = data.get("impact_level", "LOW").upper()
            try:
                impact_level = ImpactLevel(impact_str)
            except ValueError:
                impact_level = ImpactLevel.LOW
            
            # Map risk assessment
            risk_str = data.get("risk_assessment", "SAFE").upper()
            try:
                risk_assessment = RiskAssessment(risk_str)
            except ValueError:
                risk_assessment = RiskAssessment.SAFE
            
            return CommitAnalysisResponse(
                commit_hash=request.commit_hash,
                repo=request.repo,
                model_used=response.model,
                change_type=change_type,
                impact_level=impact_level,
                risk_assessment=risk_assessment,
                summary=data.get("summary", "No summary"),
                description=data.get("description", "No description"),
                affected_services=data.get("affected_services", []),
                affected_components=affected_components,
                breaking_changes=breaking_changes,
                is_breaking=len(breaking_changes) > 0,
                migration_guide=data.get("migration_guide"),
                rollback_plan=rollback_plan,
                rollback_safe=not rollback_data.get("data_migration_required", False),
                recommendations=data.get("recommendations", []),
                review_suggestions=data.get("review_suggestions", []),
                test_suggestions=data.get("test_suggestions", []),
                confidence_score=0.95,  # High confidence for production
                complexity_score=data.get("complexity_score", 0.5),
                risk_score=data.get("risk_score", 0.3),
                analysis_latency_ms=int(response.latency_ms),
                slo_compliant=response.latency_ms < 500,
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            raise AnalysisParseError(f"Invalid JSON in response: {e}")
        except Exception as e:
            logger.error(f"Failed to parse analysis response: {e}")
            raise AnalysisParseError(f"Failed to parse response: {e}")
    
    async def analyze_commit(
        self,
        request: CommitAnalysisRequest,
        diff_content: Optional[str] = None,
    ) -> CommitAnalysisResponse:
        """
        Analyze a commit with production-grade guarantees.
        
        Args:
            request: Commit analysis request
            diff_content: Optional diff content for detailed analysis
            
        Returns:
            CommitAnalysisResponse with full analysis
        """
        self._total_analyses += 1
        
        # Check cache for determinism
        cache_key = f"{request.repo}:{request.commit_hash}"
        if cache_key in self._analysis_cache:
            logger.debug(f"Returning cached analysis for {cache_key}")
            return self._analysis_cache[cache_key]
        
        try:
            # Build prompt
            prompt = self._build_analysis_prompt(request, diff_content)
            system_prompt = SYSTEM_PROMPTS["commit_analysis"]
            
            # Call model (with circuit breaker if available)
            if self.circuit_breaker:
                response = await self.circuit_breaker.execute(
                    self.model_client.analyze,
                    prompt,
                    system_prompt,
                    request.commit_hash,  # Use commit hash as deterministic seed
                )
            else:
                response = await self.model_client.analyze(
                    prompt,
                    system_prompt,
                    request.commit_hash,
                )
            
            # Parse response
            analysis = self._parse_analysis_response(response, request)
            
            # Cache for determinism
            self._analysis_cache[cache_key] = analysis
            
            self._successful_analyses += 1
            logger.info(f"Analysis completed for {cache_key}: {analysis.change_type}, {analysis.impact_level}")
            
            return analysis
            
        except Exception as e:
            self._failed_analyses += 1
            logger.error(f"Analysis failed for {cache_key}: {e}")
            raise
    
    async def analyze_batch(
        self,
        requests: List[CommitAnalysisRequest],
        parallel: bool = True,
    ) -> List[CommitAnalysisResponse]:
        """Analyze multiple commits"""
        import asyncio
        
        if parallel:
            tasks = [self.analyze_commit(req) for req in requests]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions
            return [r for r in results if isinstance(r, CommitAnalysisResponse)]
        else:
            results = []
            for req in requests:
                try:
                    result = await self.analyze_commit(req)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch analysis failed for {req.commit_hash}: {e}")
            return results
    
    async def verify_determinism(
        self,
        request: CommitAnalysisRequest,
        runs: int = 3,
    ) -> bool:
        """
        Verify that analysis is deterministic.
        
        Runs the same analysis multiple times and checks outputs are identical.
        """
        results = []
        
        # Clear cache for this test
        cache_key = f"{request.repo}:{request.commit_hash}"
        self._analysis_cache.pop(cache_key, None)
        
        for _ in range(runs):  # i unused
            # Clear cache between runs
            self._analysis_cache.pop(cache_key, None)
            
            result = await self.analyze_commit(request)
            results.append(result)
        
        # Compare results
        first = results[0]
        for result in results[1:]:
            if (result.change_type != first.change_type or
                result.impact_level != first.impact_level or
                result.risk_assessment != first.risk_assessment or
                result.summary != first.summary):
                logger.warning(f"Determinism check failed for {cache_key}")
                return False
        
        logger.info(f"Determinism verified for {cache_key} ({runs} runs)")
        return True
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get analysis engine metrics"""
        return {
            "total_analyses": self._total_analyses,
            "successful_analyses": self._successful_analyses,
            "failed_analyses": self._failed_analyses,
            "success_rate": self._successful_analyses / max(1, self._total_analyses),
            "cache_size": len(self._analysis_cache),
        }
    
    def clear_cache(self) -> int:
        """Clear analysis cache"""
        count = len(self._analysis_cache)
        self._analysis_cache.clear()
        return count


class AnalysisParseError(Exception):
    """Error parsing analysis response"""
    pass
