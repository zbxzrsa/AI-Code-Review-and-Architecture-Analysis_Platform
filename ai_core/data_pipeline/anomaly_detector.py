"""
Anomaly Detection and Auto-Repair System
Intelligent detection and correction of data anomalies
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
import logging

logger = logging.getLogger(__name__)


class AnomalyType(Enum):
    """Types of anomalies"""
    OUTLIER = "outlier"
    MISSING = "missing"
    DUPLICATE = "duplicate"
    FORMAT = "format"
    RANGE = "range"
    TYPE = "type"
    INCONSISTENT = "inconsistent"


@dataclass
class Anomaly:
    """Detected anomaly"""
    index: int
    column: Optional[str]
    anomaly_type: AnomalyType
    value: Any
    score: float
    suggested_fix: Optional[Any]
    confidence: float


class AnomalyDetector:
    """
    Intelligent Anomaly Detection System
    
    Features:
    - Multiple detection algorithms
    - Ensemble detection
    - Contextual anomaly detection
    - Automatic threshold tuning
    """
    
    def __init__(
        self,
        method: str = 'ensemble',
        contamination: float = 0.1,
        sensitivity: float = 0.5
    ):
        """
        Initialize Anomaly Detector
        
        Args:
            method: 'isolation_forest', 'lof', 'dbscan', or 'ensemble'
            contamination: Expected proportion of anomalies
            sensitivity: Detection sensitivity (0-1)
        """
        self.method = method
        self.contamination = contamination
        self.sensitivity = sensitivity
        
        self.detectors = self._initialize_detectors()
    
    def _initialize_detectors(self) -> Dict[str, Any]:
        """Initialize detection algorithms"""
        return {
            'isolation_forest': IsolationForest(
                contamination=self.contamination,
                random_state=42,
                n_estimators=100
            ),
            'lof': LocalOutlierFactor(
                contamination=self.contamination,
                novelty=False
            ),
            'dbscan': DBSCAN(eps=0.5, min_samples=5)
        }
    
    def detect(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        column: Optional[str] = None
    ) -> List[Anomaly]:
        """
        Detect anomalies in data
        
        Args:
            data: Input data
            column: Specific column to analyze (for DataFrame)
            
        Returns:
            List of detected anomalies
        """
        if isinstance(data, pd.DataFrame):
            if column:
                return self._detect_column(data[column].values, column)
            else:
                return self._detect_dataframe(data)
        else:
            return self._detect_array(data)
    
    def _detect_array(
        self,
        data: np.ndarray,
        column: Optional[str] = None
    ) -> List[Anomaly]:
        """Detect anomalies in a numpy array"""
        anomalies = []
        
        # Handle 1D data
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Remove NaN for detection
        valid_mask = ~np.isnan(data).any(axis=1)
        valid_data = data[valid_mask]
        
        if len(valid_data) < 10:
            return anomalies
        
        if self.method == 'ensemble':
            scores = self._ensemble_detect(valid_data)
        elif self.method == 'isolation_forest':
            scores = self._isolation_forest_detect(valid_data)
        elif self.method == 'lof':
            scores = self._lof_detect(valid_data)
        elif self.method == 'dbscan':
            scores = self._dbscan_detect(valid_data)
        else:
            scores = self._ensemble_detect(valid_data)
        
        # Map scores back to original indices
        full_scores = np.zeros(len(data))
        full_scores[valid_mask] = scores
        
        # Identify anomalies based on threshold
        threshold = np.percentile(
            scores, 
            100 * (1 - self.contamination * self.sensitivity)
        )
        
        for idx, score in enumerate(full_scores):
            if score > threshold or np.isnan(data[idx]).any():
                anomaly_type = (
                    AnomalyType.MISSING if np.isnan(data[idx]).any()
                    else AnomalyType.OUTLIER
                )
                
                anomalies.append(Anomaly(
                    index=idx,
                    column=column,
                    anomaly_type=anomaly_type,
                    value=data[idx].tolist() if data.ndim > 1 else data[idx],
                    score=float(score),
                    suggested_fix=None,
                    confidence=min(1.0, score / threshold) if threshold > 0 else 0.5
                ))
        
        return anomalies
    
    def _detect_column(
        self,
        data: np.ndarray,
        column: str
    ) -> List[Anomaly]:
        """Detect anomalies in a single column"""
        return self._detect_array(data, column)
    
    def _detect_dataframe(self, df: pd.DataFrame) -> List[Anomaly]:
        """Detect anomalies in a DataFrame"""
        all_anomalies = []
        
        # Numerical columns
        for col in df.select_dtypes(include=[np.number]).columns:
            anomalies = self._detect_column(df[col].values, col)
            all_anomalies.extend(anomalies)
        
        # Categorical/text columns
        for col in df.select_dtypes(include=['object']).columns:
            anomalies = self._detect_categorical_anomalies(df[col], col)
            all_anomalies.extend(anomalies)
        
        # Duplicate detection
        duplicates = df[df.duplicated(keep=False)]
        for idx in duplicates.index:
            all_anomalies.append(Anomaly(
                index=idx,
                column=None,
                anomaly_type=AnomalyType.DUPLICATE,
                value=df.loc[idx].tolist(),
                score=1.0,
                suggested_fix='Remove duplicate',
                confidence=1.0
            ))
        
        return all_anomalies
    
    def _detect_categorical_anomalies(
        self,
        series: pd.Series,
        column: str
    ) -> List[Anomaly]:
        """Detect anomalies in categorical data"""
        anomalies = []
        
        # Find rare values
        value_counts = series.value_counts(normalize=True)
        threshold = self.contamination * self.sensitivity
        
        rare_values = value_counts[value_counts < threshold].index.tolist()
        
        for idx, value in series.items():
            if pd.isna(value):
                anomalies.append(Anomaly(
                    index=idx,
                    column=column,
                    anomaly_type=AnomalyType.MISSING,
                    value=value,
                    score=1.0,
                    suggested_fix=value_counts.index[0] if len(value_counts) > 0 else None,
                    confidence=1.0
                ))
            elif value in rare_values:
                anomalies.append(Anomaly(
                    index=idx,
                    column=column,
                    anomaly_type=AnomalyType.OUTLIER,
                    value=value,
                    score=1.0 - value_counts.get(value, 0),
                    suggested_fix=value_counts.index[0] if len(value_counts) > 0 else None,
                    confidence=0.7
                ))
        
        return anomalies
    
    def _isolation_forest_detect(self, data: np.ndarray) -> np.ndarray:
        """Isolation Forest detection"""
        self.detectors['isolation_forest'].fit(data)
        scores = -self.detectors['isolation_forest'].score_samples(data)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def _lof_detect(self, data: np.ndarray) -> np.ndarray:
        """Local Outlier Factor detection"""
        self.detectors['lof'].fit(data)
        scores = -self.detectors['lof'].negative_outlier_factor_
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    
    def _dbscan_detect(self, data: np.ndarray) -> np.ndarray:
        """DBSCAN-based detection"""
        labels = self.detectors['dbscan'].fit_predict(data)
        # Points labeled -1 are noise/outliers
        scores = np.where(labels == -1, 1.0, 0.0)
        return scores
    
    def _ensemble_detect(self, data: np.ndarray) -> np.ndarray:
        """Ensemble detection combining multiple methods"""
        scores = []
        
        # Isolation Forest
        if_scores = self._isolation_forest_detect(data)
        scores.append(if_scores)
        
        # LOF
        if len(data) >= 5:  # LOF needs minimum samples
            lof_scores = self._lof_detect(data)
            scores.append(lof_scores)
        
        # Statistical (z-score based)
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0) + 1e-8
        z_scores = np.abs((data - mean) / std)
        z_max = np.max(z_scores, axis=1)
        z_normalized = (z_max - z_max.min()) / (z_max.max() - z_max.min() + 1e-8)
        scores.append(z_normalized)
        
        # Combine scores
        combined = np.mean(np.stack(scores), axis=0)
        return combined


class AutoRepair:
    """
    Automatic Data Repair System
    
    Features:
    - Multiple repair strategies
    - Context-aware correction
    - Confidence-based repair
    - Rollback support
    """
    
    def __init__(
        self,
        repair_threshold: float = 0.7,
        max_repairs_per_column: float = 0.1
    ):
        """
        Initialize Auto-Repair
        
        Args:
            repair_threshold: Minimum confidence to apply repair
            max_repairs_per_column: Maximum proportion of values to repair
        """
        self.repair_threshold = repair_threshold
        self.max_repairs = max_repairs_per_column
        
        self.repair_log: List[Dict[str, Any]] = []
    
    def repair(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        anomalies: List[Anomaly],
        strategy: str = 'auto'
    ) -> Tuple[Union[np.ndarray, pd.DataFrame], List[Dict[str, Any]]]:
        """
        Repair detected anomalies
        
        Args:
            data: Data to repair
            anomalies: List of detected anomalies
            strategy: 'auto', 'mean', 'median', 'mode', 'drop', 'interpolate'
            
        Returns:
            Repaired data and repair log
        """
        repairs = []
        
        if isinstance(data, pd.DataFrame):
            repaired = data.copy()
        else:
            repaired = data.copy()
        
        # Group anomalies by column
        anomalies_by_column: Dict[str, List[Anomaly]] = {}
        for anomaly in anomalies:
            col = anomaly.column or '_default'
            if col not in anomalies_by_column:
                anomalies_by_column[col] = []
            anomalies_by_column[col].append(anomaly)
        
        # Repair each column
        for column, col_anomalies in anomalies_by_column.items():
            # Check repair limit
            if isinstance(data, pd.DataFrame) and column != '_default':
                max_allowed = int(len(data) * self.max_repairs)
                col_anomalies = sorted(
                    col_anomalies,
                    key=lambda a: a.confidence,
                    reverse=True
                )[:max_allowed]
            
            for anomaly in col_anomalies:
                if anomaly.confidence < self.repair_threshold:
                    continue
                
                repair_info = self._apply_repair(
                    repaired, anomaly, strategy
                )
                
                if repair_info:
                    repairs.append(repair_info)
        
        self.repair_log.extend(repairs)
        
        return repaired, repairs
    
    def _apply_repair(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        anomaly: Anomaly,
        strategy: str
    ) -> Optional[Dict[str, Any]]:
        """Apply a single repair"""
        repair_info = {
            'index': anomaly.index,
            'column': anomaly.column,
            'original_value': anomaly.value,
            'anomaly_type': anomaly.anomaly_type.value,
            'strategy': strategy
        }
        
        try:
            if isinstance(data, pd.DataFrame):
                new_value = self._compute_repair_value(
                    data[anomaly.column] if anomaly.column else data.iloc[:, 0],
                    anomaly,
                    strategy
                )
                
                if anomaly.column:
                    data.loc[anomaly.index, anomaly.column] = new_value
                else:
                    data.iloc[anomaly.index] = new_value
            else:
                new_value = self._compute_repair_value(
                    data[:, 0] if data.ndim > 1 else data,
                    anomaly,
                    strategy
                )
                data[anomaly.index] = new_value
            
            repair_info['new_value'] = new_value
            repair_info['success'] = True
            
        except Exception as e:
            repair_info['error'] = str(e)
            repair_info['success'] = False
        
        return repair_info
    
    def _compute_repair_value(
        self,
        column_data: Union[np.ndarray, pd.Series],
        anomaly: Anomaly,
        strategy: str
    ) -> Any:
        """Compute the repair value based on strategy"""
        if anomaly.suggested_fix is not None and strategy == 'auto':
            return anomaly.suggested_fix
        
        # Get valid values (excluding the anomaly)
        if isinstance(column_data, pd.Series):
            valid = column_data.drop(anomaly.index).dropna()
        else:
            mask = np.ones(len(column_data), dtype=bool)
            mask[anomaly.index] = False
            valid = column_data[mask]
            valid = valid[~np.isnan(valid)]
        
        if len(valid) == 0:
            return 0 if np.issubdtype(type(anomaly.value), np.number) else ''
        
        if strategy == 'auto':
            # Choose based on anomaly type
            if anomaly.anomaly_type == AnomalyType.MISSING:
                strategy = 'median'
            elif anomaly.anomaly_type == AnomalyType.OUTLIER:
                strategy = 'median'
            else:
                strategy = 'mode'
        
        if strategy == 'mean':
            return float(np.mean(valid))
        elif strategy == 'median':
            return float(np.median(valid))
        elif strategy == 'mode':
            if isinstance(valid, pd.Series):
                return valid.mode().iloc[0] if len(valid.mode()) > 0 else valid.iloc[0]
            else:
                values, counts = np.unique(valid, return_counts=True)
                return values[np.argmax(counts)]
        elif strategy == 'interpolate':
            # Linear interpolation
            idx = anomaly.index
            if idx > 0 and idx < len(column_data) - 1:
                return (column_data[idx-1] + column_data[idx+1]) / 2
            elif idx == 0:
                return column_data[1]
            else:
                return column_data[-2]
        else:
            return float(np.median(valid))
    
    def get_repair_summary(self) -> Dict[str, Any]:
        """Get summary of all repairs"""
        if not self.repair_log:
            return {'total_repairs': 0}
        
        summary = {
            'total_repairs': len(self.repair_log),
            'successful': sum(1 for r in self.repair_log if r.get('success', False)),
            'failed': sum(1 for r in self.repair_log if not r.get('success', True)),
            'by_type': {},
            'by_strategy': {}
        }
        
        for repair in self.repair_log:
            atype = repair.get('anomaly_type', 'unknown')
            strategy = repair.get('strategy', 'unknown')
            
            summary['by_type'][atype] = summary['by_type'].get(atype, 0) + 1
            summary['by_strategy'][strategy] = summary['by_strategy'].get(strategy, 0) + 1
        
        return summary
    
    def rollback(
        self,
        data: Union[np.ndarray, pd.DataFrame],
        n_repairs: int = 1
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Rollback the last n repairs"""
        if not self.repair_log:
            return data
        
        repairs_to_rollback = self.repair_log[-n_repairs:]
        
        for repair in reversed(repairs_to_rollback):
            if repair.get('success', False):
                if isinstance(data, pd.DataFrame):
                    if repair['column']:
                        data.loc[repair['index'], repair['column']] = repair['original_value']
                    else:
                        data.iloc[repair['index']] = repair['original_value']
                else:
                    data[repair['index']] = repair['original_value']
        
        self.repair_log = self.repair_log[:-n_repairs]
        
        return data
