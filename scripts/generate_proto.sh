#!/usr/bin/env bash
# Generate gRPC Python code from proto files

set -e

PROTO_DIR="$(dirname "$0")/../proto"
OUT_DIR="$(dirname "$0")/../src/docuflow_ml/proto"

echo "Generating gRPC Python code from proto files..."
echo "Proto dir: $PROTO_DIR"
echo "Output dir: $OUT_DIR"

mkdir -p "$OUT_DIR"

# Generate Python gRPC code
python -m grpc_tools.protoc \
    -I"$PROTO_DIR" \
    --python_out="$OUT_DIR" \
    --grpc_python_out="$OUT_DIR" \
    --proto_path="$PROTO_DIR" \
    "$PROTO_DIR"/docuflow/v1/document.proto

# Fix imports in generated files
echo "Fixing imports..."
find "$OUT_DIR" -name "*.py" -exec sed -i 's/^import docuflow\./from . import docuflow./' {} \;
find "$OUT_DIR" -name "*.py" -exec sed -i 's/^from docuflow\./from . docuflow./' {} \;

echo "Generation complete!"