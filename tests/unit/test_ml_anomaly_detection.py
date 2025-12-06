"""
Unit tests for ML Anomaly Detection System.

Tests cover:
- Feature extraction
- ML model training and prediction
- Detection rules and alerts
- Model persistence
"""
import asyncio
import numpy as np
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from backend.shared.security.ml_anomaly_detection import (
    FeatureExtractor,
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    AutoencoderDetector,
    EnsembleDetector,
    AnomalyDetectionService,
    AnomalyAlert,
    DetectionRule,
    ModelMetrics,
    AnomalyType,
    SeverityLevel,
    ModelStatus
)


class TestFeatureExtractor:
    """Tests for FeatureExtractor class."""
    
    def test_feature_extraction_basic(self):
        """Test basic feature extraction."""
        extractor = FeatureExtractor()
        
        log_entry = {
            "id": "log1",
            "entity": "version",
            "action": "promote",
            "actor_id": "user1",
            "status": "success",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        context = {
            "actor_history": {
                "actions_per_hour": 10,
                "unique_entities": 5,
                "unique_resources": 20,
                "action_diversity": 0.5,
                "entity_diversity": 0.3
            },
            "volume": {
                "log_count_1h": 100,
                "log_count_24h": 1000,
                "burst_ratio": 1.5
            },
            "time_since_last": 60
        }
        
        features = extractor.extract_features(log_entry, context)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0
    
    def test_feature_extraction_batch(self):
        """Test batch feature extraction."""
        extractor = FeatureExtractor()
        
        log_entries = [
            {
                "id": f"log{i}",
                "entity": "version",
                "action": "promote",
                "actor_id": "user1",
                "status": "success",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            for i in range(10)
        ]
        
        contexts = [
            {"actor_history": {}, "volume": {}, "time_since_last": 0}
            for _ in range(10)
        ]
        
        features = extractor.extract_batch_features(log_entries, contexts)
        
        assert features.shape[0] == 10
        assert features.ndim == 2
    
    def test_action_encoding(self):
        """Test action string encoding."""
        extractor = FeatureExtractor()
        
        code1 = extractor._encode_action("promote")
        code2 = extractor._encode_action("promote")  # Same action
        code3 = extractor._encode_action("quarantine")  # Different action
        
        assert code1 == code2
        assert code1 != code3
    
    def test_entity_encoding(self):
        """Test entity string encoding."""
        extractor = FeatureExtractor()
        
        code1 = extractor._encode_entity("version")
        code2 = extractor._encode_entity("version")
        code3 = extractor._encode_entity("experiment")
        
        assert code1 == code2
        assert code1 != code3
    
    def test_temporal_features(self):
        """Test temporal feature extraction."""
        extractor = FeatureExtractor()
        
        # Weekday business hours
        log_entry = {
            "timestamp": "2024-01-15T10:30:00+00:00",  # Monday 10:30
            "action": "test",
            "entity": "test"
        }
        
        features = extractor.extract_features(log_entry, {"actor_history": {}, "volume": {}})
        
        # Hour feature should be normalized
        assert 0 <= features[0] <= 1
    
    def test_empty_context(self):
        """Test feature extraction with empty context."""
        extractor = FeatureExtractor()
        
        log_entry = {
            "id": "log1",
            "entity": "version",
            "action": "test",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        features = extractor.extract_features(log_entry, {})
        assert len(features) > 0


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = IsolationForestDetector(
            n_estimators=50,
            contamination=0.05
        )
        
        assert detector.n_estimators == 50
        assert detector.contamination == 0.05
    
    def test_fit_and_predict(self):
        """Test training and prediction."""
        detector = IsolationForestDetector()
        
        # Generate normal data
        np.random.seed(42)
        X_normal = np.random.randn(100, 10)
        
        detector.fit(X_normal)
        assert detector._is_fitted
        
        # Predict on normal data
        scores = detector.predict(X_normal[:10])
        assert len(scores) == 10
        assert all(0 <= s <= 1 for s in scores)
    
    def test_predict_without_fit(self):
        """Test prediction without fitting raises error."""
        detector = IsolationForestDetector()
        
        with pytest.raises(RuntimeError):
            detector.predict(np.random.randn(5, 10))
    
    def test_anomaly_detection(self):
        """Test that anomalies get higher scores."""
        detector = IsolationForestDetector()
        
        # Normal data
        np.random.seed(42)
        X_normal = np.random.randn(200, 5)
        
        detector.fit(X_normal)
        
        # Test normal point
        normal_scores = detector.predict(np.zeros((1, 5)))
        
        # Test anomalous point (far from normal)
        anomaly_scores = detector.predict(np.ones((1, 5)) * 10)
        
        # Anomaly should have higher score (more anomalous)
        assert anomaly_scores[0] >= normal_scores[0] or np.isclose(anomaly_scores[0], normal_scores[0])


class TestLocalOutlierFactorDetector:
    """Tests for LocalOutlierFactorDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = LocalOutlierFactorDetector(
            n_neighbors=15,
            contamination=0.1
        )
        
        assert detector.n_neighbors == 15
        assert detector.contamination == 0.1
    
    def test_fit_and_predict(self):
        """Test training and prediction."""
        detector = LocalOutlierFactorDetector()
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        detector.fit(X)
        assert detector._is_fitted
        
        scores = detector.predict(X[:10])
        assert len(scores) == 10


class TestAutoencoderDetector:
    """Tests for AutoencoderDetector class."""
    
    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = AutoencoderDetector(
            encoding_dim=4,
            hidden_dims=[16, 8],
            epochs=10
        )
        
        assert detector.encoding_dim == 4
        assert detector.hidden_dims == [16, 8]
    
    def test_fit_and_predict(self):
        """Test training and prediction."""
        detector = AutoencoderDetector(epochs=5)  # Fewer epochs for testing
        
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        detector.fit(X)
        assert detector._is_fitted
        
        scores = detector.predict(X[:10])
        assert len(scores) == 10
        assert all(0 <= s <= 1 for s in scores)


class TestEnsembleDetector:
    """Tests for EnsembleDetector class."""
    
    def test_ensemble_creation(self):
        """Test ensemble creation."""
        detector = EnsembleDetector()
        
        assert len(detector.detectors) == 3  # Default 3 detectors
        assert len(detector.weights) == 3
    
    def test_custom_ensemble(self):
        """Test ensemble with custom detectors."""
        custom_detectors = [
            IsolationForestDetector(),
            LocalOutlierFactorDetector()
        ]
        
        detector = EnsembleDetector(
            detectors=custom_detectors,
            weights=[0.6, 0.4]
        )
        
        assert len(detector.detectors) == 2
        assert detector.weights == [0.6, 0.4]
    
    def test_ensemble_fit_predict(self):
        """Test ensemble training and prediction."""
        detector = EnsembleDetector(detectors=[
            IsolationForestDetector(),
            LocalOutlierFactorDetector()
        ])
        
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        detector.fit(X)
        assert detector._is_fitted
        
        scores = detector.predict(X[:10])
        assert len(scores) == 10


class TestAnomalyAlert:
    """Tests for AnomalyAlert dataclass."""
    
    def test_alert_creation(self):
        """Test creating an alert."""
        alert = AnomalyAlert(
            alert_id="alert1",
            anomaly_type=AnomalyType.BEHAVIORAL,
            severity=SeverityLevel.HIGH,
            confidence=0.85,
            description="Test alert",
            affected_logs=["log1", "log2"],
            features={"score": 0.85},
            timestamp=datetime.now(timezone.utc)
        )
        
        assert alert.alert_id == "alert1"
        assert not alert.acknowledged
        assert not alert.resolved
    
    def test_alert_to_dict(self):
        """Test alert serialization."""
        alert = AnomalyAlert(
            alert_id="alert1",
            anomaly_type=AnomalyType.VOLUMETRIC,
            severity=SeverityLevel.CRITICAL,
            confidence=0.95,
            description="High volume",
            affected_logs=["log1"],
            features={},
            timestamp=datetime.now(timezone.utc)
        )
        
        d = alert.to_dict()
        assert d["alert_id"] == "alert1"
        assert d["anomaly_type"] == "volumetric"
        assert d["severity"] == "critical"


class TestDetectionRule:
    """Tests for DetectionRule dataclass."""
    
    def test_rule_creation(self):
        """Test creating a detection rule."""
        rule = DetectionRule(
            rule_id="R001",
            name="Test Rule",
            anomaly_type=AnomalyType.TEMPORAL,
            condition="hour < 6 or hour > 22",
            threshold=0.7,
            severity=SeverityLevel.MEDIUM
        )
        
        assert rule.enabled is True
        assert rule.cooldown_seconds == 300
    
    def test_rule_to_dict(self):
        """Test rule serialization."""
        rule = DetectionRule(
            rule_id="R001",
            name="Test Rule",
            anomaly_type=AnomalyType.TEMPORAL,
            condition="test",
            threshold=0.5,
            severity=SeverityLevel.LOW
        )
        
        d = rule.to_dict()
        assert d["rule_id"] == "R001"
        assert d["enabled"] is True


class TestAnomalyDetectionService:
    """Tests for AnomalyDetectionService class."""
    
    @pytest.fixture
    def mock_db(self):
        """Create mock database client."""
        db = AsyncMock()
        db.execute = AsyncMock()
        db.fetch = AsyncMock(return_value=[])
        db.fetchone = AsyncMock(return_value=None)
        return db
    
    def test_service_initialization(self, mock_db):
        """Test service initialization."""
        service = AnomalyDetectionService(
            db_client=mock_db,
            detector_type="isolation_forest",
            anomaly_threshold=0.8
        )
        
        assert service.anomaly_threshold == 0.8
        assert len(service.rules) > 0  # Default rules
    
    def test_add_rule(self, mock_db):
        """Test adding detection rule."""
        service = AnomalyDetectionService(db_client=mock_db)
        
        new_rule = DetectionRule(
            rule_id="R100",
            name="Custom Rule",
            anomaly_type=AnomalyType.BEHAVIORAL,
            condition="test",
            threshold=0.6,
            severity=SeverityLevel.HIGH
        )
        
        service.add_rule(new_rule)
        assert "R100" in service.rules
    
    def test_remove_rule(self, mock_db):
        """Test removing detection rule."""
        service = AnomalyDetectionService(db_client=mock_db)
        initial_count = len(service.rules)
        
        service.remove_rule("R001")  # Remove default rule
        assert len(service.rules) == initial_count - 1
    
    def test_acknowledge_alert(self, mock_db):
        """Test acknowledging an alert."""
        service = AnomalyDetectionService(db_client=mock_db)
        
        alert = AnomalyAlert(
            alert_id="alert1",
            anomaly_type=AnomalyType.BEHAVIORAL,
            severity=SeverityLevel.HIGH,
            confidence=0.85,
            description="Test",
            affected_logs=[],
            features={},
            timestamp=datetime.now(timezone.utc)
        )
        service.alerts.append(alert)
        
        result = service.acknowledge_alert("alert1")
        assert result is True
        assert alert.acknowledged is True
    
    def test_resolve_alert(self, mock_db):
        """Test resolving an alert."""
        service = AnomalyDetectionService(db_client=mock_db)
        
        alert = AnomalyAlert(
            alert_id="alert1",
            anomaly_type=AnomalyType.BEHAVIORAL,
            severity=SeverityLevel.HIGH,
            confidence=0.85,
            description="Test",
            affected_logs=[],
            features={},
            timestamp=datetime.now(timezone.utc)
        )
        service.alerts.append(alert)
        
        result = service.resolve_alert("alert1", "Fixed the issue")
        assert result is True
        assert alert.resolved is True
        assert alert.resolution_notes == "Fixed the issue"
    
    def test_get_alerts_filtered(self, mock_db):
        """Test getting filtered alerts."""
        service = AnomalyDetectionService(db_client=mock_db)
        
        # Add alerts with different severities
        for i, severity in enumerate([SeverityLevel.LOW, SeverityLevel.HIGH, SeverityLevel.CRITICAL]):
            alert = AnomalyAlert(
                alert_id=f"alert{i}",
                anomaly_type=AnomalyType.BEHAVIORAL,
                severity=severity,
                confidence=0.85,
                description="Test",
                affected_logs=[],
                features={},
                timestamp=datetime.now(timezone.utc)
            )
            service.alerts.append(alert)
        
        # Filter by severity
        high_alerts = service.get_alerts(severity=SeverityLevel.HIGH)
        assert len(high_alerts) == 1
    
    def test_get_status(self, mock_db):
        """Test getting service status."""
        service = AnomalyDetectionService(db_client=mock_db)
        
        status = service.get_status()
        
        assert "model_status" in status
        assert "total_rules" in status
        assert "total_alerts" in status
    
    def test_score_to_severity(self, mock_db):
        """Test score to severity conversion."""
        service = AnomalyDetectionService(db_client=mock_db)
        
        assert service._score_to_severity(0.96) == SeverityLevel.CRITICAL
        assert service._score_to_severity(0.90) == SeverityLevel.HIGH
        assert service._score_to_severity(0.80) == SeverityLevel.MEDIUM
        assert service._score_to_severity(0.70) == SeverityLevel.LOW
    
    def test_evaluate_rule(self, mock_db):
        """Test rule evaluation."""
        service = AnomalyDetectionService(db_client=mock_db)
        
        rule = DetectionRule(
            rule_id="test",
            name="Test",
            anomaly_type=AnomalyType.BEHAVIORAL,
            condition="action == 'delete'",
            threshold=0.5,
            severity=SeverityLevel.HIGH
        )
        
        log_entry = {"action": "delete", "entity": "version"}
        context = {}
        
        result = service._evaluate_rule(rule, log_entry, context)
        assert result is True
        
        log_entry["action"] = "create"
        result = service._evaluate_rule(rule, log_entry, context)
        assert result is False


class TestModelMetrics:
    """Tests for ModelMetrics dataclass."""
    
    def test_metrics_creation(self):
        """Test creating model metrics."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            false_positive_rate=0.05,
            training_samples=1000,
            last_trained=datetime.now(timezone.utc)
        )
        
        assert metrics.accuracy == 0.95
    
    def test_metrics_to_dict(self):
        """Test metrics serialization."""
        metrics = ModelMetrics(
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            false_positive_rate=0.05,
            training_samples=1000,
            last_trained=datetime.now(timezone.utc)
        )
        
        d = metrics.to_dict()
        assert d["accuracy"] == 0.95
        assert "last_trained" in d


class TestAnomalyTypes:
    """Tests for enum types."""
    
    def test_anomaly_type_values(self):
        """Test anomaly type enum values."""
        assert AnomalyType.TEMPORAL.value == "temporal"
        assert AnomalyType.BEHAVIORAL.value == "behavioral"
        assert AnomalyType.VOLUMETRIC.value == "volumetric"
    
    def test_severity_level_values(self):
        """Test severity level enum values."""
        assert SeverityLevel.LOW.value == "low"
        assert SeverityLevel.CRITICAL.value == "critical"
    
    def test_model_status_values(self):
        """Test model status enum values."""
        assert ModelStatus.TRAINING.value == "training"
        assert ModelStatus.READY.value == "ready"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
