"""
Data Quality Assessment System
Automated quality evaluation and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import Counter
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class QualityDimension(Enum):
    """Quality dimensions for assessment"""
    COMPLETENESS = "completeness"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    TIMELINESS = "timeliness"
    UNIQUENESS = "uniqueness"
    VALIDITY = "validity"


@dataclass
class DimensionScore:
    """Score for a quality dimension"""
    dimension: QualityDimension
    score: float  # 0.0 to 1.0
    issues: List[str]
    recommendations: List[str]


@dataclass
class DataQualityReport:
    """Comprehensive data quality report"""
    overall_score: float
    dimension_scores: Dict[str, DimensionScore]
    data_profile: Dict[str, Any]
    issues_summary: List[str]
    recommendations: List[str]
    timestamp: str
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'overall_score': self.overall_score,
            'dimension_scores': {
                k: {
                    'dimension': v.dimension.value,
                    'score': v.score,
                    'issues': v.issues,
                    'recommendations': v.recommendations
                }
                for k, v in self.dimension_scores.items()
            },
            'data_profile': self.data_profile,
            'issues_summary': self.issues_summary,
            'recommendations': self.recommendations,
            'timestamp': self.timestamp
        }
    
    def save(self, path: str) -> None:
        """Save report to JSON file"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


class QualityAssessor:
    """
    Data Quality Assessment System
    
    Features:
    - Multi-dimensional quality scoring
    - Automated profiling
    - Issue detection
    - Recommendations generation
    - Trend analysis
    """
    
    def __init__(
        self,
        completeness_threshold: float = 0.95,
        uniqueness_threshold: float = 0.99,
        validity_rules: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Quality Assessor
        
        Args:
            completeness_threshold: Minimum completeness score
            uniqueness_threshold: Minimum uniqueness score
            validity_rules: Custom validity rules per column
        """
        self.completeness_threshold = completeness_threshold
        self.uniqueness_threshold = uniqueness_threshold
        self.validity_rules = validity_rules or {}
        
        self.assessment_history: List[DataQualityReport] = []
    
    def assess(
        self,
        data: Union[pd.DataFrame, np.ndarray, List],
        data_type: str = 'auto'
    ) -> DataQualityReport:
        """
        Assess data quality
        
        Args:
            data: Data to assess
            data_type: 'dataframe', 'array', 'text', or 'auto'
            
        Returns:
            DataQualityReport
        """
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            if data.ndim == 1:
                df = pd.DataFrame({'values': data})
            else:
                df = pd.DataFrame(data)
            data_type = 'array'
        elif isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                df = pd.DataFrame({'text': data})
                data_type = 'text'
            else:
                df = pd.DataFrame({'values': data})
        else:
            df = data
            data_type = 'dataframe'
        
        # Profile the data
        profile = self._profile_data(df, data_type)
        
        # Assess each dimension
        dimension_scores = {}
        
        # Completeness
        completeness = self._assess_completeness(df)
        dimension_scores['completeness'] = completeness
        
        # Uniqueness
        uniqueness = self._assess_uniqueness(df)
        dimension_scores['uniqueness'] = uniqueness
        
        # Validity
        validity = self._assess_validity(df, data_type)
        dimension_scores['validity'] = validity
        
        # Consistency
        consistency = self._assess_consistency(df)
        dimension_scores['consistency'] = consistency
        
        # Calculate overall score
        weights = {
            'completeness': 0.3,
            'uniqueness': 0.2,
            'validity': 0.3,
            'consistency': 0.2
        }
        
        overall_score = sum(
            dimension_scores[dim].score * weight
            for dim, weight in weights.items()
        )
        
        # Compile issues and recommendations
        all_issues = []
        all_recommendations = []
        
        for dim_score in dimension_scores.values():
            all_issues.extend(dim_score.issues)
            all_recommendations.extend(dim_score.recommendations)
        
        report = DataQualityReport(
            overall_score=overall_score,
            dimension_scores=dimension_scores,
            data_profile=profile,
            issues_summary=all_issues,
            recommendations=list(set(all_recommendations)),
            timestamp=pd.Timestamp.now().isoformat()
        )
        
        self.assessment_history.append(report)
        
        return report
    
    def _profile_data(
        self,
        df: pd.DataFrame,
        data_type: str
    ) -> Dict[str, Any]:
        """Generate data profile"""
        profile = {
            'row_count': len(df),
            'column_count': len(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
            'data_type': data_type
        }
        
        # Column profiles
        column_profiles = {}
        for col in df.columns:
            col_profile = {
                'dtype': str(df[col].dtype),
                'null_count': df[col].isnull().sum(),
                'null_percentage': df[col].isnull().mean() * 100,
                'unique_count': df[col].nunique(),
                'unique_percentage': df[col].nunique() / len(df) * 100
            }
            
            # Numerical statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                col_profile.update({
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'median': df[col].median()
                })
            
            # Text statistics
            elif pd.api.types.is_string_dtype(df[col]):
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    col_profile.update({
                        'avg_length': non_null.str.len().mean(),
                        'min_length': non_null.str.len().min(),
                        'max_length': non_null.str.len().max()
                    })
            
            column_profiles[col] = col_profile
        
        profile['columns'] = column_profiles
        
        return profile
    
    def _assess_completeness(self, df: pd.DataFrame) -> DimensionScore:
        """Assess data completeness"""
        issues = []
        recommendations = []
        
        # Calculate completeness score
        total_cells = df.size
        null_cells = df.isnull().sum().sum()
        score = 1.0 - (null_cells / total_cells) if total_cells > 0 else 0.0
        
        # Identify columns with missing values
        for col in df.columns:
            null_pct = df[col].isnull().mean()
            if null_pct > 0:
                issues.append(f"Column '{col}' has {null_pct*100:.1f}% missing values")
                
                if null_pct > 0.5:
                    recommendations.append(f"Consider dropping column '{col}' (>50% missing)")
                elif null_pct > 0.1:
                    recommendations.append(f"Impute missing values in column '{col}'")
        
        if score < self.completeness_threshold:
            recommendations.append(
                f"Overall completeness ({score*100:.1f}%) below threshold "
                f"({self.completeness_threshold*100:.1f}%)"
            )
        
        return DimensionScore(
            dimension=QualityDimension.COMPLETENESS,
            score=score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> DimensionScore:
        """Assess data uniqueness"""
        issues = []
        recommendations = []
        
        # Check for duplicate rows
        duplicate_count = df.duplicated().sum()
        duplicate_pct = duplicate_count / len(df) if len(df) > 0 else 0
        
        score = 1.0 - duplicate_pct
        
        if duplicate_count > 0:
            issues.append(f"{duplicate_count} duplicate rows ({duplicate_pct*100:.1f}%)")
            recommendations.append("Remove duplicate rows")
        
        # Check individual columns
        for col in df.columns:
            unique_pct = df[col].nunique() / len(df) if len(df) > 0 else 0
            if unique_pct < 0.01 and len(df) > 100:  # Very low uniqueness
                issues.append(f"Column '{col}' has very low uniqueness ({unique_pct*100:.2f}%)")
        
        return DimensionScore(
            dimension=QualityDimension.UNIQUENESS,
            score=score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _assess_validity(
        self,
        df: pd.DataFrame,
        data_type: str = "generic"  # noqa: ARG002 - reserved for type-specific validation
    ) -> DimensionScore:
        """Assess data validity"""
        issues = []
        validity_scores = []
        
        for col in df.columns:
            col_score, col_issues = self._assess_column_validity(df, col)
            if col_score is not None:
                validity_scores.append(col_score)
                issues.extend(col_issues)
        
        score = np.mean(validity_scores) if validity_scores else 1.0
        recommendations = ["Review and fix invalid values"] if issues else []
        
        return DimensionScore(
            dimension=QualityDimension.VALIDITY,
            score=score,
            issues=issues,
            recommendations=recommendations
        )
    
    def _assess_column_validity(
        self,
        df: pd.DataFrame,
        col: str
    ) -> Tuple[Optional[float], List[str]]:
        """Assess validity of a single column"""
        issues = []
        col_issues = 0
        total_values = len(df[col].dropna())
        
        if total_values == 0:
            return None, []
        
        # Check against validity rules
        col_issues += self._check_validity_rules(df, col, issues)
        
        # Type-specific validation
        col_issues += self._check_numeric_validity(df, col, issues)
        
        col_score = 1.0 - (col_issues / total_values)
        return col_score, issues
    
    def _check_validity_rules(
        self,
        df: pd.DataFrame,
        col: str,
        issues: List[str]
    ) -> int:
        """Check column against validity rules"""
        col_issues = 0
        if col not in self.validity_rules:
            return 0
        
        rule = self.validity_rules[col]
        
        if 'min' in rule:
            invalid = (df[col] < rule['min']).sum()
            if invalid > 0:
                issues.append(f"{invalid} values in '{col}' below minimum")
                col_issues += invalid
        
        if 'max' in rule:
            invalid = (df[col] > rule['max']).sum()
            if invalid > 0:
                issues.append(f"{invalid} values in '{col}' above maximum")
                col_issues += invalid
        
        if 'pattern' in rule:
            import re
            pattern = re.compile(rule['pattern'])
            invalid = (~df[col].astype(str).str.match(pattern)).sum()
            if invalid > 0:
                issues.append(f"{invalid} values in '{col}' don't match pattern")
                col_issues += invalid
        
        return col_issues
    
    def _check_numeric_validity(
        self,
        df: pd.DataFrame,
        col: str,
        issues: List[str]
    ) -> int:
        """Check numeric column for infinity values"""
        if not pd.api.types.is_numeric_dtype(df[col]):
            return 0
        
        inf_count = np.isinf(df[col].dropna()).sum()
        if inf_count > 0:
            issues.append(f"{inf_count} infinite values in '{col}'")
        return inf_count
    
    def _assess_consistency(self, df: pd.DataFrame) -> DimensionScore:
        """Assess data consistency"""
        issues = []
        recommendations = []
        consistency_scores = []
        
        # Check format consistency within columns
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                # Check for mixed types
                types = df[col].dropna().apply(type).unique()
                if len(types) > 1:
                    issues.append(f"Column '{col}' has mixed types")
                    consistency_scores.append(0.5)
                else:
                    consistency_scores.append(1.0)
        
        # Check for inconsistent capitalization in text columns
        for col in df.select_dtypes(include=['object']).columns:
            non_null = df[col].dropna()
            if len(non_null) > 0:
                lower_ratio = (non_null.str.lower() == non_null).mean()
                upper_ratio = (non_null.str.upper() == non_null).mean()
                
                if 0.1 < lower_ratio < 0.9 and 0.1 < upper_ratio < 0.9:
                    issues.append(f"Column '{col}' has inconsistent capitalization")
                    recommendations.append(f"Standardize capitalization in '{col}'")
        
        score = np.mean(consistency_scores) if consistency_scores else 1.0
        
        return DimensionScore(
            dimension=QualityDimension.CONSISTENCY,
            score=score,
            issues=issues,
            recommendations=recommendations
        )
    
    def compare_reports(
        self,
        report_a: DataQualityReport,
        report_b: DataQualityReport
    ) -> Dict[str, Any]:
        """Compare two quality reports"""
        comparison = {
            'overall_score_change': report_b.overall_score - report_a.overall_score,
            'dimension_changes': {},
            'new_issues': [],
            'resolved_issues': []
        }
        
        for dim in report_a.dimension_scores:
            if dim in report_b.dimension_scores:
                change = (
                    report_b.dimension_scores[dim].score -
                    report_a.dimension_scores[dim].score
                )
                comparison['dimension_changes'][dim] = change
        
        # Find new and resolved issues
        issues_a = set(report_a.issues_summary)
        issues_b = set(report_b.issues_summary)
        
        comparison['new_issues'] = list(issues_b - issues_a)
        comparison['resolved_issues'] = list(issues_a - issues_b)
        
        return comparison
    
    def get_trend(self, dimension: Optional[str] = None) -> Dict[str, List[float]]:
        """Get quality score trends over time"""
        if not self.assessment_history:
            return {}
        
        if dimension:
            return {
                dimension: [
                    r.dimension_scores.get(dimension, DimensionScore(
                        QualityDimension.COMPLETENESS, 0, [], []
                    )).score
                    for r in self.assessment_history
                ]
            }
        
        trends = {'overall': [r.overall_score for r in self.assessment_history]}
        
        for dim in ['completeness', 'uniqueness', 'validity', 'consistency']:
            trends[dim] = [
                r.dimension_scores.get(dim, DimensionScore(
                    QualityDimension.COMPLETENESS, 0, [], []
                )).score
                for r in self.assessment_history
            ]
        
        return trends
    
    def generate_summary(self) -> str:
        """Generate a human-readable summary"""
        if not self.assessment_history:
            return "No assessments performed yet."
        
        latest = self.assessment_history[-1]
        
        summary = f"""
Data Quality Assessment Summary
==============================
Timestamp: {latest.timestamp}

Overall Score: {latest.overall_score:.2%}

Dimension Scores:
"""
        
        for dim, score in latest.dimension_scores.items():
            summary += f"  - {dim.capitalize()}: {score.score:.2%}\n"
        
        summary += "\nData Profile:\n"
        summary += f"  - Rows: {latest.data_profile.get('row_count', 'N/A')}\n"
        summary += f"  - Columns: {latest.data_profile.get('column_count', 'N/A')}\n"
        
        if latest.issues_summary:
            summary += f"\nTop Issues ({len(latest.issues_summary)}):\n"
            for issue in latest.issues_summary[:5]:
                summary += f"  - {issue}\n"
        
        if latest.recommendations:
            summary += f"\nRecommendations ({len(latest.recommendations)}):\n"
            for rec in latest.recommendations[:5]:
                summary += f"  - {rec}\n"
        
        return summary
