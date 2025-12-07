"""
Data Cleaning Pipeline

Orchestrates the full cleaning process:
1. Format normalization
2. Quality assessment
3. Duplicate detection
4. V2 system integration
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

import aiohttp

from ..collectors.base import CollectedItem
from ..config import NetworkedLearningConfig, QualityThresholds
from .deduplication import DuplicateDetector, DuplicateGroup
from .normalizer import FormatNormalizer
from .quality import QualityAssessor, QualityScore

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """
    Result of cleaning pipeline execution.
    
    Attributes:
        input_count: Number of items input
        output_count: Number of items output (passed all checks)
        filtered_quality: Items filtered by quality
        filtered_duplicate: Items filtered as duplicates
        processing_time_ms: Time taken in milliseconds
        pushed_to_v2: Number of items pushed to V2
        errors: List of error messages
    """
    input_count: int = 0
    output_count: int = 0
    filtered_quality: int = 0
    filtered_duplicate: int = 0
    processing_time_ms: float = 0.0
    pushed_to_v2: int = 0
    errors: List[str] = field(default_factory=list)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_count": self.input_count,
            "output_count": self.output_count,
            "filtered_quality": self.filtered_quality,
            "filtered_duplicate": self.filtered_duplicate,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "pushed_to_v2": self.pushed_to_v2,
            "errors": self.errors,
            "pass_rate": round(self.output_count / max(self.input_count, 1) * 100, 1),
        }


class DataCleaningPipeline:
    """
    Data cleaning pipeline for collected items.
    
    Pipeline stages:
    1. Normalize: Standardize format
    2. Assess: Calculate quality scores
    3. Filter: Remove low quality items
    4. Dedupe: Remove duplicates
    5. Push: Send to V2 system
    
    Quality threshold: >= 0.8
    Duplicate threshold: >= 0.85 similarity
    """
    
    def __init__(
        self,
        config: NetworkedLearningConfig,
        on_item_processed: Optional[Callable[[CollectedItem, QualityScore], None]] = None,
    ):
        self.config = config
        self.on_item_processed = on_item_processed
        
        # Pipeline components
        self.normalizer = FormatNormalizer()
        self.quality_assessor = QualityAssessor(config.quality)
        self.duplicate_detector = DuplicateDetector(
            similarity_threshold=config.quality.duplicate_similarity_threshold
        )
        
        # V2 integration
        self._session: Optional[aiohttp.ClientSession] = None
        self._v2_endpoint = config.v2_push_endpoint
        self._v2_batch_size = config.v2_batch_size
    
    async def start(self):
        """Initialize pipeline resources."""
        self._session = aiohttp.ClientSession()
        logger.info("Cleaning pipeline started")
    
    async def stop(self):
        """Cleanup pipeline resources."""
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("Cleaning pipeline stopped")
    
    async def process(
        self,
        items: List[CollectedItem],
        push_to_v2: bool = True,
    ) -> CleaningResult:
        """
        Process items through cleaning pipeline.
        
        Args:
            items: Items to process
            push_to_v2: Whether to push cleaned items to V2
            
        Returns:
            CleaningResult with statistics
        """
        import time
        start_time = time.time()
        
        result = CleaningResult(input_count=len(items))
        cleaned_items: List[CollectedItem] = []
        
        try:
            # Stage 1: Normalize
            normalized_items = [self.normalizer.normalize(item) for item in items]
            
            # Stage 2 & 3: Assess and Filter
            quality_passed = []
            for item in normalized_items:
                score = self.quality_assessor.assess(item)
                result.quality_scores[item.unique_id] = score.overall
                
                if self.on_item_processed:
                    self.on_item_processed(item, score)
                
                if score.passed:
                    quality_passed.append(item)
                else:
                    result.filtered_quality += 1
                    logger.debug(
                        f"Quality filter: {item.unique_id} "
                        f"(score={score.overall:.2f}, reasons={score.reasons})"
                    )
            
            # Stage 4: Deduplicate
            unique_items, duplicate_groups = self.duplicate_detector.deduplicate_batch(
                quality_passed
            )
            result.filtered_duplicate = sum(len(g.duplicate_ids) for g in duplicate_groups)
            
            cleaned_items = unique_items
            result.output_count = len(cleaned_items)
            
            # Stage 5: Push to V2
            if push_to_v2 and cleaned_items:
                pushed = await self._push_to_v2(cleaned_items)
                result.pushed_to_v2 = pushed
            
        except Exception as e:
            result.errors.append(str(e))
            logger.error(f"Pipeline error: {e}")
        
        result.processing_time_ms = (time.time() - start_time) * 1000
        
        logger.info(
            f"Cleaning complete: {result.input_count} in, "
            f"{result.output_count} out, "
            f"{result.filtered_quality} quality filtered, "
            f"{result.filtered_duplicate} duplicates"
        )
        
        return result
    
    async def process_stream(
        self,
        items: List[CollectedItem],
    ):
        """
        Process items as a stream, yielding cleaned items.
        
        Args:
            items: Items to process
            
        Yields:
            Cleaned items that pass all checks
        """
        for item in items:
            # Normalize
            normalized = self.normalizer.normalize(item)
            
            # Assess quality
            score = self.quality_assessor.assess(normalized)
            
            if not score.passed:
                continue
            
            # Check duplicate
            existing = self.duplicate_detector.find_duplicate(normalized)
            if existing:
                continue
            
            # Add to index and yield
            self.duplicate_detector.add_to_index(normalized)
            yield normalized
    
    async def _push_to_v2(self, items: List[CollectedItem]) -> int:
        """
        Push cleaned items to V2 system.
        
        Args:
            items: Cleaned items to push
            
        Returns:
            Number of items successfully pushed
        """
        if not self._session:
            await self.start()
        
        pushed = 0
        
        # Process in batches
        for i in range(0, len(items), self._v2_batch_size):
            batch = items[i:i + self._v2_batch_size]
            
            payload = {
                "items": [item.to_dict() for item in batch],
                "source": "networked_learning",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            
            try:
                async with self._session.post(
                    self._v2_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as response:
                    if response.status == 200:
                        pushed += len(batch)
                    else:
                        logger.warning(
                            f"V2 push failed: {response.status} - "
                            f"{await response.text()}"
                        )
            except asyncio.TimeoutError:
                logger.error("V2 push timeout")
            except aiohttp.ClientError as e:
                logger.error(f"V2 push error: {e}")
            except Exception as e:
                logger.error(f"Unexpected V2 push error: {e}")
        
        return pushed
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "duplicate_index_size": self.duplicate_detector.index_size,
            "quality_threshold": self.config.quality.min_quality_score,
            "duplicate_threshold": self.config.quality.duplicate_similarity_threshold,
        }
    
    def reset(self):
        """Reset pipeline state."""
        self.duplicate_detector.clear_index()
        logger.info("Pipeline state reset")
