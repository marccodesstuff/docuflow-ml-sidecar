"""Main entry point for ML sidecar service."""

import logging
import sys
from pathlib import Path

from docuflow_ml.config import settings
from docuflow_ml.grpc_service import serve

# Setup logging
if settings.log_format == "json":
    import pythonjsonlogger.jsonlogger
    handler = logging.StreamHandler()
    handler.setFormatter(pythonjsonlogger.jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    ))
    logging.root.handlers = [handler]
else:
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

logger = logging.getLogger(__name__)


def main():
    """Main entry point."""
    logger.info(f"Starting DocuFlow ML Sidecar v{settings.__version__}")
    logger.info(f"Configuration: host={settings.host}, port={settings.port}")
    logger.info(f"Device: {settings.get_device()}, dtype: {settings.get_torch_dtype()}")
    
    # Pre-load models on startup (optional, can be lazy)
    # model_manager.load_classifier_model()
    # model_manager.load_donut_model()
    # model_manager.load_table_detection_model()
    
    # Start gRPC server
    serve()


if __name__ == "__main__":
    main()