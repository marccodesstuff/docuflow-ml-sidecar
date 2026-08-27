"""DocuFlow ML Sidecar - gRPC service for document understanding."""

import os
import sys
from pathlib import Path

# Add proto directory to path for imports
PROTO_DIR = Path(__file__).parent.parent / "proto"
sys.path.insert(0, str(PROTO_DIR))

__version__ = "1.0.0"