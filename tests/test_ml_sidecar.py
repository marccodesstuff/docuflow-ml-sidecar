"""Tests for ML sidecar."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image
import io

from docuflow_ml.config import Settings
from docuflow_ml.models import ModelManager


class TestSettings:
    """Test configuration settings."""
    
    def test_default_settings(self):
        settings = Settings()
        assert settings.host == "0.0.0.0"
        assert settings.port == 50051
        assert settings.device == "auto"
    
    def test_device_resolution_cpu(self):
        settings = Settings(device="cpu")
        assert settings.get_device() == "cpu"
    
    def test_device_resolution_cuda(self):
        settings = Settings(device="cuda")
        assert settings.get_device() == "cuda"


class TestModelManager:
    """Test model manager."""
    
    @pytest.fixture
    def mock_settings(self):
        with patch("docuflow_ml.models.settings") as mock:
            mock.get_device.return_value = "cpu"
            mock.get_torch_dtype.return_value = "float32"
            mock.model_cache_dir = "/tmp/models"
            mock.layoutlm_classifier_model = "microsoft/layoutlmv3-base"
            mock.layoutlm_extraction_model = "microsoft/layoutlmv3-base"
            mock.donut_model = "naver-clova-ix/donut-base-finetuned-cord-v2"
            mock.table_detection_model = "microsoft/table-transformer-detection"
            mock.table_structure_model = "microsoft/table-transformer-structure"
            mock.element_detection_model = "microsoft/layoutlmv3-base"
            mock.max_image_size = 4096
            mock.max_sequence_length = 512
            mock.confidence_threshold = 0.5
            yield mock
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        img = Image.new("RGB", (800, 600), color="white")
        return img
    
    def test_preprocess_image_resize(self, mock_settings, sample_image):
        manager = ModelManager()
        # Create large image
        large_img = Image.new("RGB", (5000, 4000), color="white")
        processed = manager.preprocess_image(large_img)
        assert max(processed.size) <= 4096
    
    def test_preprocess_image_rgb_conversion(self, mock_settings, sample_image):
        manager = ModelManager()
        # Create grayscale image
        gray_img = Image.new("L", (100, 100), color=128)
        processed = manager.preprocess_image(gray_img)
        assert processed.mode == "RGB"
    
    def test_health_check(self, mock_settings):
        manager = ModelManager()
        health = manager.health_check()
        assert health["healthy"] is True
        assert "device" in health
        assert "loaded_models" in health


class TestGRPCService:
    """Test gRPC service (mocked)."""
    
    @pytest.fixture
    def mock_model_manager(self):
        with patch("docuflow_ml.grpc_service.model_manager") as mock:
            mock.classify_document.return_value = {
                "predicted_type": "invoice",
                "confidence": 0.95,
                "all_scores": {"invoice": 0.95, "receipt": 0.05},
            }
            mock.extract_fields.return_value = {
                "fields": {"total": {"value": "100.00", "confidence": 0.9}},
                "overall_confidence": 0.9,
                "model_version": "test-model",
            }
            mock.detect_tables.return_value = [
                {"bbox": {"x": 0, "y": 0, "width": 100, "height": 100}, "confidence": 0.8}
            ]
            mock.detect_elements.return_value = []
            mock.health_check.return_value = {
                "healthy": True,
                "device": "cpu",
                "loaded_models": [],
            }
            yield mock
    
    def test_service_initialization(self, mock_model_manager):
        from docuflow_ml.grpc_service import MLInferenceServicer
        servicer = MLInferenceServicer()
        assert servicer.model_manager is not None


# Integration test markers
pytestmark = pytest.mark.asyncio