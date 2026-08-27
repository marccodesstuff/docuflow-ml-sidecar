# DocuFlow ML Sidecar

Python gRPC service for document understanding using LayoutLM, Donut, and Table Transformer models.

## Features

- **Document Classification**: LayoutLMv3 for document type classification
- **Field Extraction**: Donut for key-value extraction from documents
- **Table Detection**: Table Transformer for table detection and structure recognition
- **Element Detection**: Checkbox, signature, and form element detection
- **gRPC API**: High-performance service communication with DocuFlow Core

## Models Used

| Task | Model | Source |
|------|-------|--------|
| Classification | LayoutLMv3-base | Microsoft |
| Field Extraction | Donut-base (CORD-v2) | Naver Clova |
| Table Detection | Table Transformer Detection | Microsoft |
| Table Structure | Table Transformer Structure | Microsoft |
| Element Detection | LayoutLMv3-base (fine-tuned) | Microsoft |

## Quick Start

### Prerequisites

- Python 3.11+
- CUDA 11.8+ (for GPU acceleration, optional)
- 8GB+ RAM (16GB+ recommended for GPU)

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# For GPU support
pip install -e ".[gpu]"
```

### Generate gRPC Code

```bash
./scripts/generate_proto.sh
```

### Run Service

```bash
# Development
python -m docuflow_ml.main

# With custom config
export DOCUFLOW_ML_PORT=50051
export DOCUFLOW_ML_DEVICE=cuda
python -m docuflow_ml.main
```

### Docker

```bash
# Build
docker build -t docuflow-ml-sidecar:latest .

# Run
docker run -d \
  -p 50051:50051 \
  -p 9090:9090 \
  -v /path/to/models:/models \
  docuflow-ml-sidecar:latest
```

## Configuration

Environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUFLOW_ML_HOST` | `0.0.0.0` | Server host |
| `DOCUFLOW_ML_PORT` | `50051` | gRPC port |
| `DOCUFLOW_ML_DEVICE` | `auto` | Device: auto, cpu, cuda, mps |
| `DOCUFLOW_ML_MODEL_CACHE_DIR` | `/models` | Model cache directory |
| `DOCUFLOW_ML_LOG_LEVEL` | `INFO` | Log level |
| `DOCUFLOW_ML_MAX_BATCH_SIZE` | `4` | Max batch size for inference |

## gRPC API

Defined in `proto/docuflow/v1/document.proto`:

- `ClassifyDocument` - Classify document type
- `ExtractFields` - Extract key-value fields
- `DetectTables` - Detect tables in document
- `DetectElements` - Detect form elements
- `HealthCheck` - Service health check

## Model Loading

Models are loaded lazily on first request and cached in memory. To pre-load models at startup, uncomment the pre-load calls in `main.py`.

## Monitoring

- Prometheus metrics on port 9090 (if enabled)
- Structured JSON logging
- Health check endpoint via gRPC

## Development

```bash
# Run tests
pytest

# Format code
ruff format .
black .

# Lint
ruff check .
mypy src/
```