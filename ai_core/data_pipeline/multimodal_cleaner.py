"""
Multi-Modal Data Cleaner
Handles text, images, and structured data
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
import re
import json
from PIL import Image
import io

logger = logging.getLogger(__name__)


@dataclass
class CleaningConfig:
    """Configuration for multi-modal cleaning"""
    # Text settings
    text_min_length: int = 10
    text_max_length: int = 10000
    remove_html: bool = True
    remove_urls: bool = True
    normalize_whitespace: bool = True
    
    # Image settings
    image_min_size: Tuple[int, int] = (32, 32)
    image_max_size: Tuple[int, int] = (4096, 4096)
    image_formats: List[str] = None
    normalize_images: bool = True
    
    # Structured data settings
    handle_missing: str = 'mean'
    handle_outliers: str = 'clip'
    outlier_std: float = 3.0
    
    def __post_init__(self):
        if self.image_formats is None:
            self.image_formats = ['JPEG', 'PNG', 'GIF', 'BMP', 'WEBP']


class TextProcessor:
    """Text data processor"""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
        self.html_pattern = re.compile(r'<[^>]+>')
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Clean a single text"""
        stats = {'original_length': len(text), 'modifications': []}
        
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Remove HTML
        if self.config.remove_html:
            new_text = self.html_pattern.sub(' ', text)
            if new_text != text:
                stats['modifications'].append('html_removed')
            text = new_text
        
        # Remove URLs
        if self.config.remove_urls:
            new_text = self.url_pattern.sub(' ', text)
            if new_text != text:
                stats['modifications'].append('urls_removed')
            text = new_text
        
        # Normalize whitespace
        if self.config.normalize_whitespace:
            text = self.whitespace_pattern.sub(' ', text).strip()
        
        stats['cleaned_length'] = len(text)
        stats['valid'] = self.config.text_min_length <= len(text) <= self.config.text_max_length
        
        return text, stats
    
    def clean_batch(self, texts: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """Clean a batch of texts"""
        cleaned = []
        all_stats = []
        
        for text in texts:
            clean_text, stats = self.clean(text)
            if stats['valid']:
                cleaned.append(clean_text)
            all_stats.append(stats)
        
        batch_stats = {
            'original_count': len(texts),
            'cleaned_count': len(cleaned),
            'removed_count': len(texts) - len(cleaned),
            'modifications': sum(len(s['modifications']) for s in all_stats)
        }
        
        return cleaned, batch_stats


class ImageProcessor:
    """Image data processor"""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
    
    def clean(
        self,
        image: Union[Image.Image, np.ndarray, bytes]
    ) -> Tuple[Optional[Image.Image], Dict[str, Any]]:
        """Clean a single image"""
        stats = {'modifications': [], 'valid': True}
        
        try:
            # Convert to PIL Image
            if isinstance(image, bytes):
                image = Image.open(io.BytesIO(image))
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            stats['original_size'] = image.size
            stats['original_format'] = image.format
            
            # Check format
            if image.format and image.format not in self.config.image_formats:
                stats['valid'] = False
                stats['reason'] = f'Invalid format: {image.format}'
                return None, stats
            
            # Check size
            w, h = image.size
            min_w, min_h = self.config.image_min_size
            max_w, max_h = self.config.image_max_size
            
            if w < min_w or h < min_h:
                stats['valid'] = False
                stats['reason'] = 'Image too small'
                return None, stats
            
            # Resize if too large
            if w > max_w or h > max_h:
                ratio = min(max_w / w, max_h / h)
                new_size = (int(w * ratio), int(h * ratio))
                image = image.resize(new_size, Image.LANCZOS)
                stats['modifications'].append('resized')
            
            # Convert to RGB if needed
            if image.mode not in ['RGB', 'L']:
                image = image.convert('RGB')
                stats['modifications'].append('converted_to_rgb')
            
            stats['final_size'] = image.size
            
            return image, stats
            
        except Exception as e:
            stats['valid'] = False
            stats['error'] = str(e)
            return None, stats
    
    def clean_batch(
        self,
        images: List[Union[Image.Image, np.ndarray, bytes]]
    ) -> Tuple[List[Image.Image], Dict[str, Any]]:
        """Clean a batch of images"""
        cleaned = []
        all_stats = []
        
        for img in images:
            clean_img, stats = self.clean(img)
            if clean_img is not None and stats['valid']:
                cleaned.append(clean_img)
            all_stats.append(stats)
        
        batch_stats = {
            'original_count': len(images),
            'cleaned_count': len(cleaned),
            'removed_count': len(images) - len(cleaned),
            'resize_count': sum(1 for s in all_stats if 'resized' in s.get('modifications', []))
        }
        
        return cleaned, batch_stats
    
    def to_tensor(
        self,
        image: Image.Image,
        normalize: bool = True
    ) -> torch.Tensor:
        """Convert image to tensor"""
        img_array = np.array(image)
        
        # Handle grayscale
        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=2)
        
        # HWC to CHW
        tensor = torch.from_numpy(img_array).permute(2, 0, 1).float()
        
        # Normalize to [0, 1]
        if normalize:
            tensor = tensor / 255.0
        
        return tensor


class StructuredDataProcessor:
    """Structured data processor"""
    
    def __init__(self, config: CleaningConfig):
        self.config = config
    
    def clean(
        self,
        data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Clean structured data"""
        stats = {
            'original_shape': data.shape,
            'modifications': [],
            'column_stats': {}
        }
        
        cleaned = data.copy()
        
        # Process each column based on type
        for col in cleaned.columns:
            col_stats = {'dtype': str(cleaned[col].dtype)}
            
            if pd.api.types.is_numeric_dtype(cleaned[col]):
                cleaned[col], col_stats = self._clean_numeric(cleaned[col])
            elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
                cleaned[col], col_stats = self._clean_datetime(cleaned[col])
            else:
                cleaned[col], col_stats = self._clean_categorical(cleaned[col])
            
            stats['column_stats'][col] = col_stats
        
        # Remove duplicate rows
        before_dedup = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        if len(cleaned) < before_dedup:
            stats['modifications'].append(f'removed_{before_dedup - len(cleaned)}_duplicates')
        
        stats['final_shape'] = cleaned.shape
        
        return cleaned, stats
    
    def _clean_numeric(
        self,
        series: pd.Series
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Clean numeric column"""
        stats = {'dtype': 'numeric', 'modifications': []}
        
        # Handle missing values
        missing_count = series.isnull().sum()
        if missing_count > 0:
            if self.config.handle_missing == 'mean':
                fill_value = series.mean()
            elif self.config.handle_missing == 'median':
                fill_value = series.median()
            elif self.config.handle_missing == 'zero':
                fill_value = 0
            else:
                fill_value = series.median()
            
            series = series.fillna(fill_value)
            stats['modifications'].append(f'filled_{missing_count}_missing')
        
        # Handle outliers
        if self.config.handle_outliers != 'none':
            mean = series.mean()
            std = series.std()
            lower = mean - self.config.outlier_std * std
            upper = mean + self.config.outlier_std * std
            
            outlier_count = ((series < lower) | (series > upper)).sum()
            
            if outlier_count > 0 and self.config.handle_outliers == 'clip':
                series = series.clip(lower, upper)
                stats['modifications'].append(f'clipped_{outlier_count}_outliers')
        
        stats['mean'] = float(series.mean())
        stats['std'] = float(series.std())
        
        return series, stats
    
    def _clean_datetime(
        self,
        series: pd.Series
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Clean datetime column"""
        stats = {'dtype': 'datetime', 'modifications': []}
        
        # Handle missing
        missing_count = series.isnull().sum()
        if missing_count > 0:
            # Forward fill for datetime
            series = series.fillna(method='ffill').fillna(method='bfill')
            stats['modifications'].append(f'filled_{missing_count}_missing')
        
        return series, stats
    
    def _clean_categorical(
        self,
        series: pd.Series
    ) -> Tuple[pd.Series, Dict[str, Any]]:
        """Clean categorical column"""
        stats = {'dtype': 'categorical', 'modifications': []}
        
        # Handle missing
        missing_count = series.isnull().sum()
        if missing_count > 0:
            mode_value = series.mode()
            if len(mode_value) > 0:
                series = series.fillna(mode_value.iloc[0])
                stats['modifications'].append(f'filled_{missing_count}_missing')
        
        # Strip whitespace
        if series.dtype == 'object':
            series = series.str.strip()
            stats['modifications'].append('stripped_whitespace')
        
        stats['unique_count'] = series.nunique()
        
        return series, stats


class MultiModalCleaner:
    """
    Multi-Modal Data Cleaning System
    
    Features:
    - Unified interface for text, images, and structured data
    - Cross-modal consistency checking
    - Automatic type detection
    - Batch processing
    """
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        """Initialize Multi-Modal Cleaner"""
        self.config = config or CleaningConfig()
        
        self.text_processor = TextProcessor(self.config)
        self.image_processor = ImageProcessor(self.config)
        self.structured_processor = StructuredDataProcessor(self.config)
        
        self.cleaning_history: List[Dict[str, Any]] = []
    
    def clean(
        self,
        data: Any,
        data_type: Optional[str] = None
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Clean data with automatic type detection
        
        Args:
            data: Input data
            data_type: 'text', 'image', 'structured', or None for auto
            
        Returns:
            Cleaned data and statistics
        """
        # Auto-detect type
        if data_type is None:
            data_type = self._detect_type(data)
        
        if data_type == 'text':
            if isinstance(data, list):
                result, stats = self.text_processor.clean_batch(data)
            else:
                result, stats = self.text_processor.clean(data)
        
        elif data_type == 'image':
            if isinstance(data, list):
                result, stats = self.image_processor.clean_batch(data)
            else:
                result, stats = self.image_processor.clean(data)
        
        elif data_type == 'structured':
            if not isinstance(data, pd.DataFrame):
                data = pd.DataFrame(data)
            result, stats = self.structured_processor.clean(data)
        
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # Record in history
        self.cleaning_history.append({
            'data_type': data_type,
            'stats': stats
        })
        
        return result, stats
    
    def _detect_type(self, data: Any) -> str:
        """Detect data type"""
        if isinstance(data, pd.DataFrame):
            return 'structured'
        elif isinstance(data, (Image.Image, bytes)):
            return 'image'
        elif isinstance(data, str):
            return 'text'
        elif isinstance(data, list):
            if len(data) > 0:
                if isinstance(data[0], str):
                    return 'text'
                elif isinstance(data[0], (Image.Image, bytes)):
                    return 'image'
        elif isinstance(data, dict):
            return 'structured'
        
        return 'structured'  # Default
    
    def clean_multimodal_dataset(
        self,
        dataset: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
        """
        Clean a multi-modal dataset
        
        Args:
            dataset: Dictionary with 'text', 'images', 'structured' keys
            
        Returns:
            Cleaned dataset and statistics per modality
        """
        cleaned = {}
        all_stats = {}
        
        if 'text' in dataset:
            cleaned['text'], all_stats['text'] = self.clean(
                dataset['text'], 'text'
            )
        
        if 'images' in dataset:
            cleaned['images'], all_stats['images'] = self.clean(
                dataset['images'], 'image'
            )
        
        if 'structured' in dataset:
            cleaned['structured'], all_stats['structured'] = self.clean(
                dataset['structured'], 'structured'
            )
        
        # Cross-modal consistency check
        consistency_issues = self._check_cross_modal_consistency(cleaned)
        all_stats['consistency'] = consistency_issues
        
        return cleaned, all_stats
    
    def _check_cross_modal_consistency(
        self,
        dataset: Dict[str, Any]
    ) -> List[str]:
        """Check consistency across modalities"""
        issues = []
        
        # Check if all modalities have same sample count
        counts = {}
        if 'text' in dataset:
            counts['text'] = len(dataset['text']) if isinstance(dataset['text'], list) else 1
        if 'images' in dataset:
            counts['images'] = len(dataset['images']) if isinstance(dataset['images'], list) else 1
        if 'structured' in dataset and isinstance(dataset['structured'], pd.DataFrame):
            counts['structured'] = len(dataset['structured'])
        
        if len(set(counts.values())) > 1:
            issues.append(f"Inconsistent sample counts: {counts}")
        
        return issues
    
    def get_summary(self) -> Dict[str, Any]:
        """Get cleaning summary"""
        if not self.cleaning_history:
            return {'status': 'No cleaning performed'}
        
        by_type = {}
        for record in self.cleaning_history:
            dtype = record['data_type']
            if dtype not in by_type:
                by_type[dtype] = []
            by_type[dtype].append(record['stats'])
        
        return {
            'total_operations': len(self.cleaning_history),
            'by_type': by_type
        }
