"""gRPC service implementation for ML inference."""

import grpc
import logging
import time
from concurrent import futures
from typing import Dict, Any, List
from PIL import Image
import io
import base64

from docuflow_ml.models import model_manager
from docuflow_ml.config import settings

# Import generated gRPC classes (will be generated from proto)
# from docuflow.v1 import document_pb2, document_pb2_grpc

logger = logging.getLogger(__name__)


class MLInferenceServicer:
    """gRPC servicer for ML inference operations."""
    
    def __init__(self):
        self.model_manager = model_manager
    
    def ClassifyDocument(self, request, context):
        """Classify document type."""
        try:
            logger.info(f"ClassifyDocument request for document: {request.document_id}")
            
            # Load image from storage path
            image = self._load_image(request.storage_path, request.mime_type)
            
            # Run classification
            result = self.model_manager.classify_document(
                image, list(request.candidate_types)
            )
            
            # Build response
            response = document_pb2.ClassifyDocumentResponse(
                predicted_type_id=result["predicted_type"],
                confidence=result["confidence"],
            )
            response.all_scores.update(result["all_scores"])
            
            return response
            
        except Exception as e:
            logger.error(f"ClassifyDocument failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return document_pb2.ClassifyDocumentResponse()
    
    def ExtractFields(self, request, context):
        """Extract fields from document."""
        try:
            logger.info(f"ExtractFields request for document: {request.document_id}")
            
            image = self._load_image(request.storage_path, request.mime_type)
            
            # Convert field schema
            field_schema = {}
            for key, fd in request.field_schema.items():
                field_schema[key] = {
                    "key": fd.key,
                    "label": fd.label,
                    "type": fd.type,
                    "required": fd.required,
                    "description": fd.description,
                }
            
            result = self.model_manager.extract_fields(image, field_schema)
            
            # Build response
            response = document_pb2.ExtractFieldsResponse(
                overall_confidence=result["overall_confidence"],
                model_version=result["model_version"],
            )
            
            for key, field_data in result["fields"].items():
                extracted_field = document_pb2.ExtractedField(
                    key=key,
                    value=field_data["value"],
                    confidence=field_data["confidence"],
                    validated=False,
                )
                if "bbox" in field_data:
                    bbox = field_data["bbox"]
                    extracted_field.bbox.CopyFrom(
                        document_pb2.BoundingBox(
                            x=bbox.get("x", 0),
                            y=bbox.get("y", 0),
                            width=bbox.get("width", 0),
                            height=bbox.get("height", 0),
                            element_type="text",
                            confidence=field_data["confidence"],
                        )
                    )
                response.fields[key].CopyFrom(extracted_field)
            
            return response
            
        except Exception as e:
            logger.error(f"ExtractFields failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return document_pb2.ExtractFieldsResponse()
    
    def DetectTables(self, request, context):
        """Detect tables in document."""
        try:
            logger.info(f"DetectTables request for document: {request.document_id}")
            
            image = self._load_image(request.storage_path, request.mime_type)
            tables = self.model_manager.detect_tables(image)
            
            response = document_pb2.DetectTablesResponse()
            
            for table in tables[:request.max_tables]:
                table_extraction = document_pb2.TableExtraction(
                    table_id=f"table_{len(response.tables)}",
                    confidence=table["confidence"],
                )
                
                bbox = table["bbox"]
                table_extraction.bbox.CopyFrom(
                    document_pb2.BoundingBox(
                        x=bbox["x"],
                        y=bbox["y"],
                        width=bbox["width"],
                        height=bbox["height"],
                        element_type="table",
                        confidence=table["confidence"],
                    )
                )
                
                response.tables.append(table_extraction)
            
            return response
            
        except Exception as e:
            logger.error(f"DetectTables failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return document_pb2.DetectTablesResponse()
    
    def DetectElements(self, request, context):
        """Detect document elements."""
        try:
            logger.info(f"DetectElements request for document: {request.document_id}")
            
            image = self._load_image(request.storage_path, request.mime_type)
            elements = self.model_manager.detect_elements(image, list(request.element_types))
            
            response = document_pb2.DetectElementsResponse()
            
            # Group by page (simplified - single page for now)
            page = document_pb2.DocumentPage(
                page_number=1,
                storage_path=request.storage_path,
                width_px=image.width,
                height_px=image.height,
            )
            
            for elem in elements:
                bbox = elem["bbox"]
                page.elements.append(
                    document_pb2.BoundingBox(
                        x=bbox["x"],
                        y=bbox["y"],
                        width=bbox["width"],
                        height=bbox["height"],
                        element_type=elem.get("label", "unknown"),
                        confidence=elem["confidence"],
                    )
                )
            
            response.pages.append(page)
            return response
            
        except Exception as e:
            logger.error(f"DetectElements failed: {e}", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return document_pb2.DetectElementsResponse()
    
    def HealthCheck(self, request, context):
        """Health check endpoint."""
        health = self.model_manager.health_check()
        
        return document_pb2.HealthCheckResponse(
            healthy=health["healthy"],
            version=settings.__version__,
            models_loaded=health["loaded_models"],
            timestamp=time.time(),
        )
    
    def _load_image(self, storage_path: str, mime_type: str) -> Image.Image:
        """Load image from storage path."""
        # In production, this would download from S3/MinIO
        # For now, assume local file or base64 encoded
        if storage_path.startswith("data:"):
            # Base64 data URL
            header, data = storage_path.split(",", 1)
            image_data = base64.b64decode(data)
            return Image.open(io.BytesIO(image_data))
        else:
            # Local file path
            return Image.open(storage_path)


def serve():
    """Start the gRPC server."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=settings.grpc_max_workers),
        options=[
            ('grpc.max_send_message_length', settings.grpc_max_message_size),
            ('grpc.max_receive_message_length', settings.grpc_max_message_size),
        ]
    )
    
    # Add servicer
    # document_pb2_grpc.add_MLInferenceServiceServicer_to_server(
    #     MLInferenceServicer(), server
    # )
    
    port = settings.port
    server.add_insecure_port(f"[::]:{port}")
    
    logger.info(f"Starting ML inference gRPC server on port {port}")
    server.start()
    
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down server...")
        server.stop(0)


if __name__ == "__main__":
    logging.basicConfig(
        level=getattr(logging, settings.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    serve()