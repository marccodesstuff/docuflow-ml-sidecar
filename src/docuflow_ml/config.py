"""Configuration settings for ML sidecar."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pathlib import Path


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server
    host: str = "0.0.0.0"
    port: int = 50051
    grpc_max_message_size: int = 50 * 1024 * 1024  # 50MB
    grpc_max_workers: int = 4

    # Model Configuration
    model_cache_dir: Path = Path("/models")
    device: str = "auto"  # auto, cpu, cuda, mps
    torch_dtype: str = "auto"  # auto, float16, float32, bfloat16
    
    # LayoutLM Models
    layoutlm_classifier_model: str = "microsoft/layoutlmv3-base"
    layoutlm_extraction_model: str = "microsoft/layoutlmv3-base"
    donut_model: str = "naver-clova-ix/donut-base-finetuned-cord-v2"
    
    # Table Detection
    table_detection_model: str = "microsoft/table-transformer-detection"
    table_structure_model: str = "microsoft/table-transformer-structure"
    
    # Element Detection
    element_detection_model: str = "microsoft/layoutlmv3-base"
    
    # Inference Settings
    max_batch_size: int = 4
    max_sequence_length: int = 512
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.3
    
    # Image Processing
    max_image_size: int = 4096
    target_dpi: int = 300
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090
    
    # Health Check
    health_check_interval: int = 30

    def get_device(self) -> str:
        """Resolve device string to actual torch device."""
        if self.device == "auto":
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"
        return self.device

    def get_torch_dtype(self):
        """Resolve torch dtype."""
        import torch
        if self.torch_dtype == "auto":
            device = self.get_device()
            if device == "cuda":
                return torch.float16
            return torch.float32
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(self.torch_dtype, torch.float32)


settings = Settings()