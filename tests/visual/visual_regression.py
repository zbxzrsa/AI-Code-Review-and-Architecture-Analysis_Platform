"""
Visual Regression Testing System (Testing Improvement Plan #4)

Provides visual testing with:
- Screenshot comparison (Applitools/Percy-style)
- UI component screenshot mechanism
- Visual difference threshold
- Baseline screenshot management

Acceptance Criteria: All UI changes must pass visual regression verification
"""
import asyncio
import base64
import hashlib
import io
import json
import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import math

import pytest

logger = logging.getLogger(__name__)


class ComparisonResult(str, Enum):
    """Result of visual comparison."""
    MATCH = "match"           # Images are identical
    SIMILAR = "similar"       # Within acceptable threshold
    DIFFERENT = "different"   # Exceeds threshold
    NEW = "new"               # No baseline exists
    ERROR = "error"           # Comparison failed


class DiffAlgorithm(str, Enum):
    """Diff algorithms for comparison."""
    PIXEL = "pixel"           # Pixel-by-pixel comparison
    PERCEPTUAL = "perceptual" # Perceptual hashing
    STRUCTURAL = "structural" # Structural similarity (SSIM)


@dataclass
class VisualTestConfig:
    """Configuration for visual testing."""
    baseline_dir: str = "tests/visual/baselines"
    diff_dir: str = "tests/visual/diffs"
    threshold_percent: float = 0.1  # 0.1% difference allowed
    ignore_antialiasing: bool = True
    ignore_colors: bool = False
    viewport_width: int = 1920
    viewport_height: int = 1080
    full_page: bool = False
    diff_algorithm: DiffAlgorithm = DiffAlgorithm.PIXEL
    fail_on_new: bool = False  # Fail if no baseline exists


@dataclass
class ScreenshotMetadata:
    """Metadata for a screenshot."""
    name: str
    url: str
    viewport: Tuple[int, int]
    timestamp: datetime
    hash: str
    file_path: str
    component: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "url": self.url,
            "viewport": list(self.viewport),
            "timestamp": self.timestamp.isoformat(),
            "hash": self.hash,
            "file_path": self.file_path,
            "component": self.component,
        }


@dataclass
class VisualDiff:
    """Result of visual comparison."""
    name: str
    result: ComparisonResult
    diff_percent: float
    baseline_path: Optional[str]
    actual_path: str
    diff_path: Optional[str] = None
    threshold: float = 0.1
    message: str = ""
    
    @property
    def passed(self) -> bool:
        return self.result in [ComparisonResult.MATCH, ComparisonResult.SIMILAR]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "result": self.result.value,
            "passed": self.passed,
            "diff_percent": round(self.diff_percent, 4),
            "threshold": self.threshold,
            "baseline_path": self.baseline_path,
            "actual_path": self.actual_path,
            "diff_path": self.diff_path,
            "message": self.message,
        }


class ImageComparator:
    """
    Compares images for visual differences.
    
    Supports multiple comparison algorithms:
    - Pixel-by-pixel comparison
    - Perceptual hashing
    - Structural similarity (SSIM)
    """
    
    def __init__(self, config: VisualTestConfig):
        self.config = config
    
    def compare(
        self,
        baseline_path: str,
        actual_path: str,
        diff_path: Optional[str] = None
    ) -> Tuple[float, Optional[bytes]]:
        """
        Compare two images.
        
        Args:
            baseline_path: Path to baseline image
            actual_path: Path to actual image
            diff_path: Optional path to save diff image
            
        Returns:
            Tuple of (diff_percent, diff_image_bytes)
        """
        try:
            # Import PIL for image processing
            from PIL import Image, ImageChops, ImageDraw
            
            baseline = Image.open(baseline_path)
            actual = Image.open(actual_path)
            
            # Ensure same size
            if baseline.size != actual.size:
                # Resize actual to match baseline
                actual = actual.resize(baseline.size, Image.Resampling.LANCZOS)
            
            # Convert to RGB for comparison
            baseline_rgb = baseline.convert("RGB")
            actual_rgb = actual.convert("RGB")
            
            if self.config.diff_algorithm == DiffAlgorithm.PIXEL:
                diff_percent, diff_image = self._pixel_compare(baseline_rgb, actual_rgb)
            elif self.config.diff_algorithm == DiffAlgorithm.PERCEPTUAL:
                diff_percent, diff_image = self._perceptual_compare(baseline_rgb, actual_rgb)
            elif self.config.diff_algorithm == DiffAlgorithm.STRUCTURAL:
                diff_percent, diff_image = self._structural_compare(baseline_rgb, actual_rgb)
            else:
                diff_percent, diff_image = self._pixel_compare(baseline_rgb, actual_rgb)
            
            # Save diff image if path provided
            if diff_path and diff_image:
                diff_image.save(diff_path)
            
            diff_bytes = None
            if diff_image:
                buffer = io.BytesIO()
                diff_image.save(buffer, format="PNG")
                diff_bytes = buffer.getvalue()
            
            return diff_percent, diff_bytes
            
        except Exception as e:
            logger.error(f"Image comparison failed: {e}")
            raise
    
    def _pixel_compare(self, img1, img2) -> Tuple[float, Any]:
        """Pixel-by-pixel comparison."""
        from PIL import Image, ImageChops
        
        diff = ImageChops.difference(img1, img2)
        
        # Calculate difference percentage
        diff_data = list(diff.getdata())
        total_pixels = len(diff_data)
        different_pixels = 0
        
        for pixel in diff_data:
            if isinstance(pixel, tuple):
                if any(c > 10 for c in pixel):  # Threshold for "different"
                    different_pixels += 1
            elif pixel > 10:
                different_pixels += 1
        
        diff_percent = (different_pixels / total_pixels) * 100
        
        # Create diff visualization
        diff_vis = diff.copy()
        diff_vis = diff_vis.point(lambda x: min(255, x * 10))  # Amplify differences
        
        return diff_percent, diff_vis
    
    def _perceptual_compare(self, img1, img2) -> Tuple[float, Any]:
        """Perceptual hash comparison."""
        hash1 = self._perceptual_hash(img1)
        hash2 = self._perceptual_hash(img2)
        
        # Hamming distance
        diff_bits = bin(hash1 ^ hash2).count('1')
        diff_percent = (diff_bits / 64) * 100  # 64-bit hash
        
        # Create diff image
        from PIL import ImageChops
        diff = ImageChops.difference(img1, img2)
        
        return diff_percent, diff
    
    def _perceptual_hash(self, img) -> int:
        """Calculate perceptual hash."""
        # Resize to 8x8
        small = img.resize((8, 8)).convert("L")
        pixels = list(small.getdata())
        avg = sum(pixels) / len(pixels)
        
        # Generate hash
        hash_value = 0
        for i, pixel in enumerate(pixels):
            if pixel > avg:
                hash_value |= (1 << i)
        
        return hash_value
    
    def _structural_compare(self, img1, img2) -> Tuple[float, Any]:
        """Structural similarity comparison (simplified SSIM)."""
        from PIL import ImageChops, ImageFilter
        
        # Apply blur to reduce noise
        img1_blur = img1.filter(ImageFilter.GaussianBlur(radius=1.5))
        img2_blur = img2.filter(ImageFilter.GaussianBlur(radius=1.5))
        
        diff = ImageChops.difference(img1_blur, img2_blur)
        diff_data = list(diff.getdata())
        
        # Calculate mean squared error
        mse = sum(
            sum(c**2 for c in (pixel if isinstance(pixel, tuple) else (pixel,)))
            for pixel in diff_data
        ) / (len(diff_data) * 3)  # 3 channels
        
        # Convert to percentage (higher MSE = more different)
        diff_percent = min(100, (math.sqrt(mse) / 255) * 100)
        
        return diff_percent, diff


class BaselineManager:
    """
    Manages visual test baselines.
    
    Features:
    - Baseline storage and retrieval
    - Version tracking
    - Automatic baseline updates
    """
    
    def __init__(self, baseline_dir: str):
        self.baseline_dir = Path(baseline_dir)
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.baseline_dir / "metadata.json"
        self._metadata: Dict[str, ScreenshotMetadata] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        """Load baseline metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                data = json.load(f)
                for name, meta in data.items():
                    self._metadata[name] = ScreenshotMetadata(
                        name=meta["name"],
                        url=meta["url"],
                        viewport=tuple(meta["viewport"]),
                        timestamp=datetime.fromisoformat(meta["timestamp"]),
                        hash=meta["hash"],
                        file_path=meta["file_path"],
                        component=meta.get("component"),
                    )
    
    def _save_metadata(self):
        """Save baseline metadata."""
        data = {name: meta.to_dict() for name, meta in self._metadata.items()}
        with open(self.metadata_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def get_baseline(self, name: str) -> Optional[str]:
        """Get baseline image path."""
        meta = self._metadata.get(name)
        if meta:
            path = self.baseline_dir / meta.file_path
            if path.exists():
                return str(path)
        return None
    
    def save_baseline(
        self,
        name: str,
        image_path: str,
        url: str,
        viewport: Tuple[int, int],
        component: Optional[str] = None
    ):
        """Save a new baseline."""
        # Calculate hash
        with open(image_path, "rb") as f:
            image_hash = hashlib.sha256(f.read()).hexdigest()[:16]
        
        # Copy to baseline directory
        filename = f"{name.replace('/', '_').replace(' ', '_')}.png"
        dest_path = self.baseline_dir / filename
        shutil.copy2(image_path, dest_path)
        
        # Save metadata
        self._metadata[name] = ScreenshotMetadata(
            name=name,
            url=url,
            viewport=viewport,
            timestamp=datetime.now(timezone.utc),
            hash=image_hash,
            file_path=filename,
            component=component,
        )
        self._save_metadata()
        
        logger.info(f"Saved baseline: {name}")
    
    def update_baseline(self, name: str, image_path: str):
        """Update existing baseline."""
        meta = self._metadata.get(name)
        if meta:
            self.save_baseline(name, image_path, meta.url, meta.viewport, meta.component)
    
    def list_baselines(self) -> List[ScreenshotMetadata]:
        """List all baselines."""
        return list(self._metadata.values())
    
    def delete_baseline(self, name: str) -> bool:
        """Delete a baseline."""
        meta = self._metadata.pop(name, None)
        if meta:
            path = self.baseline_dir / meta.file_path
            if path.exists():
                path.unlink()
            self._save_metadata()
            return True
        return False


class VisualRegressionTester:
    """
    Main visual regression testing class.
    
    Features:
    - Screenshot capture
    - Baseline comparison
    - Diff generation
    - Report generation
    """
    
    def __init__(self, config: Optional[VisualTestConfig] = None):
        self.config = config or VisualTestConfig()
        self.baseline_manager = BaselineManager(self.config.baseline_dir)
        self.comparator = ImageComparator(self.config)
        self.results: List[VisualDiff] = []
        
        # Ensure diff directory exists
        Path(self.config.diff_dir).mkdir(parents=True, exist_ok=True)
    
    async def capture_screenshot(
        self,
        page,  # Playwright page
        name: str,
        url: Optional[str] = None,
        selector: Optional[str] = None,
        full_page: Optional[bool] = None
    ) -> str:
        """
        Capture screenshot using Playwright.
        
        Args:
            page: Playwright page object
            name: Screenshot name
            url: Optional URL to navigate to
            selector: Optional CSS selector for element screenshot
            full_page: Override full_page setting
            
        Returns:
            Path to captured screenshot
        """
        if url:
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
        
        # Set viewport
        await page.set_viewport_size({
            "width": self.config.viewport_width,
            "height": self.config.viewport_height,
        })
        
        # Capture screenshot
        screenshot_path = Path(self.config.diff_dir) / f"{name}_actual.png"
        
        if selector:
            element = await page.query_selector(selector)
            if element:
                await element.screenshot(path=str(screenshot_path))
            else:
                raise ValueError(f"Element not found: {selector}")
        else:
            await page.screenshot(
                path=str(screenshot_path),
                full_page=full_page if full_page is not None else self.config.full_page,
            )
        
        return str(screenshot_path)
    
    def compare_screenshot(
        self,
        name: str,
        actual_path: str,
        url: str = "",
        component: Optional[str] = None
    ) -> VisualDiff:
        """
        Compare screenshot against baseline.
        
        Args:
            name: Screenshot name
            actual_path: Path to actual screenshot
            url: URL of the page
            component: Optional component name
            
        Returns:
            VisualDiff result
        """
        baseline_path = self.baseline_manager.get_baseline(name)
        
        # No baseline exists
        if not baseline_path:
            if self.config.fail_on_new:
                result = VisualDiff(
                    name=name,
                    result=ComparisonResult.NEW,
                    diff_percent=100.0,
                    baseline_path=None,
                    actual_path=actual_path,
                    threshold=self.config.threshold_percent,
                    message="No baseline exists for this screenshot",
                )
            else:
                # Save as new baseline
                self.baseline_manager.save_baseline(
                    name, actual_path, url,
                    (self.config.viewport_width, self.config.viewport_height),
                    component
                )
                result = VisualDiff(
                    name=name,
                    result=ComparisonResult.NEW,
                    diff_percent=0.0,
                    baseline_path=self.baseline_manager.get_baseline(name),
                    actual_path=actual_path,
                    threshold=self.config.threshold_percent,
                    message="New baseline created",
                )
        else:
            # Compare with baseline
            try:
                diff_path = str(Path(self.config.diff_dir) / f"{name}_diff.png")
                diff_percent, _ = self.comparator.compare(baseline_path, actual_path, diff_path)
                
                if diff_percent == 0:
                    result_type = ComparisonResult.MATCH
                    message = "Screenshots are identical"
                elif diff_percent <= self.config.threshold_percent:
                    result_type = ComparisonResult.SIMILAR
                    message = f"Within threshold ({diff_percent:.4f}% <= {self.config.threshold_percent}%)"
                else:
                    result_type = ComparisonResult.DIFFERENT
                    message = f"Exceeds threshold ({diff_percent:.4f}% > {self.config.threshold_percent}%)"
                
                result = VisualDiff(
                    name=name,
                    result=result_type,
                    diff_percent=diff_percent,
                    baseline_path=baseline_path,
                    actual_path=actual_path,
                    diff_path=diff_path if result_type == ComparisonResult.DIFFERENT else None,
                    threshold=self.config.threshold_percent,
                    message=message,
                )
            except Exception as e:
                result = VisualDiff(
                    name=name,
                    result=ComparisonResult.ERROR,
                    diff_percent=100.0,
                    baseline_path=baseline_path,
                    actual_path=actual_path,
                    threshold=self.config.threshold_percent,
                    message=f"Comparison error: {str(e)}",
                )
        
        self.results.append(result)
        return result
    
    def update_baseline(self, name: str, actual_path: str):
        """Update baseline with actual screenshot."""
        self.baseline_manager.update_baseline(name, actual_path)
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate visual regression report."""
        passed = sum(1 for r in self.results if r.passed)
        failed = sum(1 for r in self.results if not r.passed)
        new_baselines = sum(1 for r in self.results if r.result == ComparisonResult.NEW)
        
        return {
            "summary": {
                "total": len(self.results),
                "passed": passed,
                "failed": failed,
                "new_baselines": new_baselines,
                "pass_rate": round(passed / len(self.results) * 100, 2) if self.results else 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "config": {
                "threshold_percent": self.config.threshold_percent,
                "viewport": [self.config.viewport_width, self.config.viewport_height],
                "diff_algorithm": self.config.diff_algorithm.value,
            },
            "results": [r.to_dict() for r in self.results],
            "passed": failed == 0,
        }
    
    def save_report(self, path: str):
        """Save report to file."""
        report = self.generate_report()
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Visual regression report saved: {path}")


# ============================================================
# Playwright Visual Test Helpers
# ============================================================

class PlaywrightVisualTest:
    """
    Helper for Playwright-based visual tests.
    
    Usage:
        async def test_homepage(page):
            visual = PlaywrightVisualTest(page)
            await visual.check("homepage", "http://localhost:3000")
    """
    
    def __init__(self, page, config: Optional[VisualTestConfig] = None):
        self.page = page
        self.tester = VisualRegressionTester(config)
    
    async def check(
        self,
        name: str,
        url: Optional[str] = None,
        selector: Optional[str] = None,
        threshold: Optional[float] = None
    ) -> VisualDiff:
        """
        Capture and compare screenshot.
        
        Args:
            name: Screenshot name
            url: Optional URL to navigate to
            selector: Optional element selector
            threshold: Optional custom threshold
            
        Returns:
            VisualDiff result
        """
        if threshold:
            self.tester.config.threshold_percent = threshold
        
        actual_path = await self.tester.capture_screenshot(
            self.page, name, url, selector
        )
        
        return self.tester.compare_screenshot(
            name, actual_path, url or self.page.url
        )
    
    async def check_component(
        self,
        name: str,
        selector: str,
        threshold: Optional[float] = None
    ) -> VisualDiff:
        """Check a specific component."""
        return await self.check(name, selector=selector, threshold=threshold)
    
    def get_results(self) -> List[VisualDiff]:
        """Get all test results."""
        return self.tester.results
    
    def assert_no_visual_changes(self):
        """Assert no visual changes detected."""
        failed = [r for r in self.tester.results if not r.passed]
        if failed:
            messages = [f"  - {r.name}: {r.message}" for r in failed]
            raise AssertionError(f"Visual regression detected:\n" + "\n".join(messages))


# ============================================================
# Test Cases
# ============================================================

@pytest.fixture
def visual_config() -> VisualTestConfig:
    """Default visual test configuration."""
    return VisualTestConfig(
        baseline_dir="tests/visual/baselines",
        diff_dir="tests/visual/diffs",
        threshold_percent=0.1,
        viewport_width=1920,
        viewport_height=1080,
    )


class TestVisualRegression:
    """Visual regression tests."""
    
    def test_image_hash_calculation(self, visual_config):
        """Test perceptual hash calculation."""
        comparator = ImageComparator(visual_config)
        
        # Create simple test image
        from PIL import Image
        img1 = Image.new("RGB", (100, 100), color="red")
        img2 = Image.new("RGB", (100, 100), color="red")
        img3 = Image.new("RGB", (100, 100), color="blue")
        
        hash1 = comparator._perceptual_hash(img1)
        hash2 = comparator._perceptual_hash(img2)
        hash3 = comparator._perceptual_hash(img3)
        
        assert hash1 == hash2  # Same image should have same hash
        assert hash1 != hash3  # Different images should have different hashes
    
    def test_pixel_comparison(self, visual_config):
        """Test pixel-by-pixel comparison."""
        from PIL import Image
        import tempfile
        
        comparator = ImageComparator(visual_config)
        
        # Create identical images
        img1 = Image.new("RGB", (100, 100), color="white")
        img2 = Image.new("RGB", (100, 100), color="white")
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f1:
            img1.save(f1.name)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f2:
                img2.save(f2.name)
                
                diff_percent, _ = comparator.compare(f1.name, f2.name)
                assert diff_percent == 0.0
    
    def test_baseline_manager(self, visual_config, tmp_path):
        """Test baseline management."""
        from PIL import Image
        
        config = VisualTestConfig(baseline_dir=str(tmp_path / "baselines"))
        manager = BaselineManager(config.baseline_dir)
        
        # Create test image
        img = Image.new("RGB", (100, 100), color="red")
        img_path = tmp_path / "test.png"
        img.save(img_path)
        
        # Save baseline
        manager.save_baseline(
            "test_page",
            str(img_path),
            "http://localhost:3000/test",
            (1920, 1080)
        )
        
        # Retrieve baseline
        baseline = manager.get_baseline("test_page")
        assert baseline is not None
        assert Path(baseline).exists()
        
        # List baselines
        baselines = manager.list_baselines()
        assert len(baselines) == 1
        assert baselines[0].name == "test_page"


# Key pages to test
VISUAL_TEST_PAGES = [
    {"name": "login", "url": "/login", "threshold": 0.1},
    {"name": "dashboard", "url": "/dashboard", "threshold": 0.5},
    {"name": "code_review", "url": "/code-review", "threshold": 0.5},
    {"name": "settings", "url": "/settings", "threshold": 0.1},
    {"name": "admin_users", "url": "/admin/users", "threshold": 0.5},
]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
