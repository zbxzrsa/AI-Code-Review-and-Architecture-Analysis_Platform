"""
数据清洗管道 (Data Cleaning Pipeline)

模块功能描述:
    自动化数据预处理和质量提升。

主要功能:
    - 文本数据清洗
    - 代码数据清洗
    - 质量评估
    - 重复数据删除

主要组件:
    - DataCleaner: 数据清洗器抽象基类
    - TextCleaner: 文本数据清洗器
    - CodeCleaner: 代码数据清洗器
    - CleaningResult: 清洗结果数据类

最后修改日期: 2024-12-07
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
import json
from abc import ABC, abstractmethod
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class CleaningResult:
    """
    清洗操作结果数据类
    
    功能描述:
        记录数据清洗操作的结果统计。
    
    属性说明:
        - original_count: 原始数据数量
        - cleaned_count: 清洗后数据数量
        - removed_count: 移除的数据数量
        - modified_count: 修改的数据数量
        - issues_found: 发现的问题列表
        - quality_improvement: 质量提升度
    """
    original_count: int
    cleaned_count: int
    removed_count: int
    modified_count: int
    issues_found: List[str]
    quality_improvement: float


class DataCleaner(ABC):
    """
    数据清洗器抽象基类
    
    功能描述:
        定义数据清洗器的基本接口。
    
    抽象方法:
        - clean(): 清洗数据并返回结果
        - validate(): 验证数据并返回问题列表
    """
    
    @abstractmethod
    def clean(self, data: Any) -> Tuple[Any, CleaningResult]:
        """
        清洗数据并返回结果
        
        参数:
            data: 要清洗的数据
        
        返回值:
            Tuple[Any, CleaningResult]: 清洗后的数据和结果统计
        """
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> List[str]:
        """
        验证数据并返回问题列表
        
        参数:
            data: 要验证的数据
        
        返回值:
            List[str]: 发现的问题列表
        """
        pass


class TextCleaner(DataCleaner):
    """
    文本数据清洗器
    
    功能描述:
        清洗和规范化文本数据。
    
    主要特性:
        - Unicode 规范化
        - HTML/XML 移除
        - 特殊字符处理
        - 空白规范化
        - 语言检测和过滤
        - 重复数据删除
    """
    
    def __init__(
        self,
        remove_html: bool = True,
        normalize_unicode: bool = True,
        remove_urls: bool = True,
        remove_emails: bool = True,
        min_length: int = 10,
        max_length: int = 10000,
        remove_duplicates: bool = True,
        lowercase: bool = False
    ):
        self.remove_html = remove_html
        self.normalize_unicode = normalize_unicode
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.lowercase = lowercase
        
        # Regex patterns
        self.html_pattern = re.compile(r'<[^>]+>')
        # Simplified URL pattern following RFC 3986
        self.url_pattern = re.compile(
            r'https?://[a-zA-Z\d\-._~:/?#\[\]@!$&\'()*+,;=%]+'
        )
        self.email_pattern = re.compile(r'[\w.-]+@[\w.-]+\.\w+')
        self.whitespace_pattern = re.compile(r'\s+')
    
    def clean_text(self, text: str) -> str:
        """Clean a single text string"""
        if not isinstance(text, str):
            return str(text) if text is not None else ""
        
        # Remove HTML
        if self.remove_html:
            text = self.html_pattern.sub(' ', text)
        
        # Remove URLs
        if self.remove_urls:
            text = self.url_pattern.sub(' ', text)
        
        # Remove emails
        if self.remove_emails:
            text = self.email_pattern.sub(' ', text)
        
        # Normalize whitespace
        text = self.whitespace_pattern.sub(' ', text).strip()
        
        # Lowercase
        if self.lowercase:
            text = text.lower()
        
        return text
    
    def clean(self, data: List[str]) -> Tuple[List[str], CleaningResult]:
        """Clean a list of text strings"""
        original_count = len(data)
        issues = []
        modified_count = 0
        
        # Clean each text
        cleaned = []
        for text in data:
            cleaned_text = self.clean_text(text)
            if cleaned_text != text:
                modified_count += 1
            cleaned.append(cleaned_text)
        
        # Filter by length
        before_filter = len(cleaned)
        cleaned = [t for t in cleaned if self.min_length <= len(t) <= self.max_length]
        if len(cleaned) < before_filter:
            issues.append(f"Removed {before_filter - len(cleaned)} texts due to length")
        
        # Remove duplicates
        if self.remove_duplicates:
            before_dedup = len(cleaned)
            cleaned = list(dict.fromkeys(cleaned))
            if len(cleaned) < before_dedup:
                issues.append(f"Removed {before_dedup - len(cleaned)} duplicate texts")
        
        result = CleaningResult(
            original_count=original_count,
            cleaned_count=len(cleaned),
            removed_count=original_count - len(cleaned),
            modified_count=modified_count,
            issues_found=issues,
            quality_improvement=self._calculate_improvement(data, cleaned)
        )
        
        return cleaned, result
    
    def validate(self, data: List[str]) -> List[str]:
        """Validate text data"""
        issues = []
        
        empty_count = sum(1 for t in data if not t or not t.strip())
        if empty_count:
            issues.append(f"{empty_count} empty texts")
        
        short_count = sum(1 for t in data if len(t) < self.min_length)
        if short_count:
            issues.append(f"{short_count} texts below minimum length")
        
        long_count = sum(1 for t in data if len(t) > self.max_length)
        if long_count:
            issues.append(f"{long_count} texts above maximum length")
        
        # Check for duplicates
        if self.remove_duplicates:
            unique_count = len(set(data))
            if unique_count < len(data):
                issues.append(f"{len(data) - unique_count} duplicate texts")
        
        return issues
    
    def _calculate_improvement(self, original: List[str], cleaned: List[str]) -> float:
        """Calculate quality improvement score"""
        if not original:
            return 0.0
        
        # Simple heuristic based on removed issues
        original_issues = len(self.validate(original))
        cleaned_issues = len(self.validate(cleaned))
        
        if original_issues == 0:
            return 1.0
        
        return 1.0 - (cleaned_issues / original_issues)


class NumericalCleaner(DataCleaner):
    """
    Numerical Data Cleaner
    
    Features:
    - Missing value handling
    - Outlier detection and handling
    - Type conversion
    - Range validation
    - Statistical normalization
    """
    
    def __init__(
        self,
        handle_missing: str = 'mean',  # 'mean', 'median', 'mode', 'drop', 'zero'
        handle_outliers: str = 'clip',  # 'clip', 'remove', 'none'
        outlier_std: float = 3.0,
        value_range: Optional[Tuple[float, float]] = None,
        normalize: bool = False
    ):
        self.handle_missing = handle_missing
        self.handle_outliers = handle_outliers
        self.outlier_std = outlier_std
        self.value_range = value_range
        self.normalize = normalize
    
    def clean(
        self,
        data: Union[np.ndarray, pd.DataFrame, List]
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], CleaningResult]:
        """Clean numerical data"""
        # Convert to numpy if needed
        if isinstance(data, list):
            data = np.array(data)
        elif isinstance(data, pd.DataFrame):
            return self._clean_dataframe(data)
        
        original_count = data.size
        issues = []
        
        # Handle missing values
        missing_mask = np.isnan(data)
        missing_count = np.sum(missing_mask)
        
        if missing_count > 0:
            issues.append(f"{missing_count} missing values")
            data = self._handle_missing(data, missing_mask)
        
        # Handle outliers
        if self.handle_outliers != 'none':
            data, outlier_count = self._handle_outliers(data)
            if outlier_count > 0:
                issues.append(f"{outlier_count} outliers")
        
        # Apply range constraints
        if self.value_range:
            min_val, max_val = self.value_range
            out_of_range = np.sum((data < min_val) | (data > max_val))
            if out_of_range > 0:
                issues.append(f"{out_of_range} values out of range")
            data = np.clip(data, min_val, max_val)
        
        # Normalize
        if self.normalize:
            data = (data - np.mean(data)) / (np.std(data) + 1e-8)
        
        result = CleaningResult(
            original_count=original_count,
            cleaned_count=data.size,
            removed_count=original_count - data.size,
            modified_count=missing_count + (outlier_count if self.handle_outliers != 'none' else 0),
            issues_found=issues,
            quality_improvement=0.0  # Calculate based on validation
        )
        
        return data, result
    
    def _handle_missing(self, data: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Handle missing values"""
        if self.handle_missing == 'drop':
            return data[~mask]
        
        fill_value = 0.0
        valid_data = data[~mask]
        
        if self.handle_missing == 'mean':
            fill_value = np.mean(valid_data)
        elif self.handle_missing == 'median':
            fill_value = np.median(valid_data)
        elif self.handle_missing == 'mode':
            fill_value = float(Counter(valid_data.tolist()).most_common(1)[0][0])
        elif self.handle_missing == 'zero':
            fill_value = 0.0
        
        data[mask] = fill_value
        return data
    
    def _handle_outliers(self, data: np.ndarray) -> Tuple[np.ndarray, int]:
        """Handle outliers"""
        mean = np.mean(data)
        std = np.std(data)
        
        lower = mean - self.outlier_std * std
        upper = mean + self.outlier_std * std
        
        outlier_mask = (data < lower) | (data > upper)
        outlier_count = np.sum(outlier_mask)
        
        if self.handle_outliers == 'clip':
            data = np.clip(data, lower, upper)
        elif self.handle_outliers == 'remove':
            data = data[~outlier_mask]
        
        return data, outlier_count
    
    def _clean_dataframe(
        self,
        df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, CleaningResult]:
        """Clean a pandas DataFrame"""
        original_count = df.size
        issues = []
        modified_count = 0
        
        # Clean each numerical column
        for col in df.select_dtypes(include=[np.number]).columns:
            col_data = df[col].values
            cleaned_col, _ = self.clean(col_data)
            
            if len(cleaned_col) == len(df):
                df[col] = cleaned_col
                modified_count += np.sum(col_data != cleaned_col)
        
        result = CleaningResult(
            original_count=original_count,
            cleaned_count=df.size,
            removed_count=0,
            modified_count=modified_count,
            issues_found=issues,
            quality_improvement=0.0
        )
        
        return df, result
    
    def validate(self, data: Union[np.ndarray, List]) -> List[str]:
        """Validate numerical data"""
        if isinstance(data, list):
            data = np.array(data)
        
        issues = []
        
        missing_count = np.sum(np.isnan(data))
        if missing_count:
            issues.append(f"{missing_count} missing values")
        
        # Check for outliers
        mean = np.nanmean(data)
        std = np.nanstd(data)
        outliers = np.sum(np.abs(data - mean) > self.outlier_std * std)
        if outliers:
            issues.append(f"{outliers} potential outliers")
        
        # Check range
        if self.value_range:
            out_of_range = np.sum(
                (data < self.value_range[0]) | (data > self.value_range[1])
            )
            if out_of_range:
                issues.append(f"{out_of_range} values out of range")
        
        return issues


class DataCleaningPipeline:
    """
    Unified Data Cleaning Pipeline
    
    Features:
    - Multi-stage cleaning
    - Customizable processing steps
    - Parallel processing
    - Audit logging
    - Rollback support
    """
    
    def __init__(
        self,
        steps: Optional[List[Tuple[str, DataCleaner]]] = None,
        parallel: bool = False,
        log_path: Optional[str] = None
    ):
        """
        Initialize Pipeline
        
        Args:
            steps: List of (name, cleaner) tuples
            parallel: Whether to process in parallel
            log_path: Path for audit logs
        """
        self.steps: List[Tuple[str, DataCleaner]] = steps or []
        self.parallel = parallel
        self.log_path = Path(log_path) if log_path else None
        
        self.history: List[Dict[str, Any]] = []
        
        if self.log_path:
            self.log_path.mkdir(parents=True, exist_ok=True)
    
    def add_step(self, name: str, cleaner: DataCleaner) -> 'DataCleaningPipeline':
        """Add a cleaning step"""
        self.steps.append((name, cleaner))
        return self
    
    def remove_step(self, name: str) -> 'DataCleaningPipeline':
        """Remove a cleaning step by name"""
        self.steps = [(n, c) for n, c in self.steps if n != name]
        return self
    
    def process(
        self,
        data: Any,
        validate_before: bool = True,
        validate_after: bool = True
    ) -> Tuple[Any, Dict[str, CleaningResult]]:
        """
        Process data through the pipeline
        
        Args:
            data: Input data
            validate_before: Whether to validate before cleaning
            validate_after: Whether to validate after cleaning
            
        Returns:
            Cleaned data and results per step
        """
        results = {}
        
        # Pre-validation
        if validate_before:
            pre_issues = self._validate_all(data)
            results['pre_validation'] = pre_issues
        
        # Process each step
        current_data = data
        for name, cleaner in self.steps:
            try:
                cleaned_data, result = cleaner.clean(current_data)
                results[name] = result
                current_data = cleaned_data
                
                logger.info(
                    f"Step '{name}': {result.original_count} -> {result.cleaned_count} "
                    f"(removed: {result.removed_count}, modified: {result.modified_count})"
                )
                
            except Exception as e:
                logger.error(f"Error in step '{name}': {e}")
                results[name] = CleaningResult(
                    original_count=0,
                    cleaned_count=0,
                    removed_count=0,
                    modified_count=0,
                    issues_found=[f"Error: {str(e)}"],
                    quality_improvement=0.0
                )
        
        # Post-validation
        if validate_after:
            post_issues = self._validate_all(current_data)
            results['post_validation'] = post_issues
        
        # Log results
        self._log_results(results)
        
        # Store in history
        self.history.append({
            'timestamp': pd.Timestamp.now().isoformat(),
            'results': {k: v if isinstance(v, list) else v.__dict__ for k, v in results.items()}
        })
        
        return current_data, results
    
    def _validate_all(self, data: Any) -> List[str]:
        """Run all validators"""
        all_issues = []
        for name, cleaner in self.steps:
            try:
                issues = cleaner.validate(data)
                all_issues.extend([f"{name}: {issue}" for issue in issues])
            except Exception as e:
                all_issues.append(f"{name}: Validation error - {e}")
        return all_issues
    
    def _log_results(self, results: Dict[str, Any]) -> None:
        """Log results to file"""
        if not self.log_path:
            return
        
        log_file = self.log_path / f"cleaning_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(log_file, 'w') as f:
            json.dump({
                k: v if isinstance(v, list) else v.__dict__
                for k, v in results.items()
            }, f, indent=2, default=str)
    
    def get_report(self) -> Dict[str, Any]:
        """Generate pipeline report"""
        if not self.history:
            return {'status': 'No processing history'}
        
        latest = self.history[-1]
        
        total_removed = sum(
            r.get('removed_count', 0) if isinstance(r, dict) else 0
            for r in latest['results'].values()
        )
        
        total_modified = sum(
            r.get('modified_count', 0) if isinstance(r, dict) else 0
            for r in latest['results'].values()
        )
        
        return {
            'timestamp': latest['timestamp'],
            'steps': len(self.steps),
            'total_removed': total_removed,
            'total_modified': total_modified,
            'step_results': latest['results']
        }
    
    @classmethod
    def create_default_text_pipeline(cls) -> 'DataCleaningPipeline':
        """Create a default text cleaning pipeline"""
        return cls([
            ('text_clean', TextCleaner(
                remove_html=True,
                normalize_unicode=True,
                remove_urls=True,
                min_length=10
            ))
        ])
    
    @classmethod
    def create_default_numerical_pipeline(cls) -> 'DataCleaningPipeline':
        """Create a default numerical cleaning pipeline"""
        return cls([
            ('numerical_clean', NumericalCleaner(
                handle_missing='mean',
                handle_outliers='clip',
                outlier_std=3.0
            ))
        ])
