"""Model manager for loading and caching ML models."""

import torch
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
    AutoModelForVision2Seq,
    DonutProcessor,
    VisionEncoderDecoderModel,
    TableTransformerForObjectDetection,
    TableTransformerForObjectDetection,
)
from PIL import Image
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import threading
from functools import lru_cache

from docuflow_ml.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading, caching, and inference for all ML models."""
    
    def __init__(self):
        self._models: Dict[str, Any] = {}
        self._processors: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._device = settings.get_device()
        self._dtype = settings.get_torch_dtype()
        logger.info(f"ModelManager initialized on device: {self._device}, dtype: {self._dtype}")
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get local path for model, downloading if necessary."""
        model_path = settings.model_cache_dir / model_name.replace("/", "_")
        model_path.mkdir(parents=True, exist_ok=True)
        return model_path
    
    def load_classifier_model(self) -> tuple:
        """Load LayoutLM classifier model."""
        with self._lock:
            if "classifier" in self._models:
                return self._models["classifier"], self._processors["classifier"]
            
            model_name = settings.layoutlm_classifier_model
            logger.info(f"Loading classifier model: {model_name}")
            
            model_path = self._get_model_path(model_name)
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_path)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                cache_dir=model_path,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)
            model.eval()
            
            self._models["classifier"] = model
            self._processors["classifier"] = processor
            logger.info("Classifier model loaded successfully")
            return model, processor
    
    def load_extraction_model(self) -> tuple:
        """Load LayoutLM extraction model."""
        with self._lock:
            if "extraction" in self._models:
                return self._models["extraction"], self._processors["extraction"]
            
            model_name = settings.layoutlm_extraction_model
            logger.info(f"Loading extraction model: {model_name}")
            
            model_path = self._get_model_path(model_name)
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_path)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                cache_dir=model_path,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)
            model.eval()
            
            self._models["extraction"] = model
            self._processors["extraction"] = processor
            logger.info("Extraction model loaded successfully")
            return model, processor
    
    def load_donut_model(self) -> tuple:
        """Load Donut model for document understanding."""
        with self._lock:
            if "donut" in self._models:
                return self._models["donut"], self._processors["donut"]
            
            model_name = settings.donut_model
            logger.info(f"Loading Donut model: {model_name}")
            
            model_path = self._get_model_path(model_name)
            processor = DonutProcessor.from_pretrained(model_name, cache_dir=model_path)
            model = VisionEncoderDecoderModel.from_pretrained(
                model_name,
                cache_dir=model_path,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)
            model.eval()
            
            self._models["donut"] = model
            self._processors["donut"] = processor
            logger.info("Donut model loaded successfully")
            return model, processor
    
    def load_table_detection_model(self) -> tuple:
        """Load Table Transformer for table detection."""
        with self._lock:
            if "table_detection" in self._models:
                return self._models["table_detection"], self._processors["table_detection"]
            
            model_name = settings.table_detection_model
            logger.info(f"Loading table detection model: {model_name}")
            
            model_path = self._get_model_path(model_name)
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_path)
            model = TableTransformerForObjectDetection.from_pretrained(
                model_name,
                cache_dir=model_path,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)
            model.eval()
            
            self._models["table_detection"] = model
            self._processors["table_detection"] = processor
            logger.info("Table detection model loaded successfully")
            return model, processor
    
    def load_table_structure_model(self) -> tuple:
        """Load Table Transformer for table structure recognition."""
        with self._lock:
            if "table_structure" in self._models:
                return self._models["table_structure"], self._processors["table_structure"]
            
            model_name = settings.table_structure_model
            logger.info(f"Loading table structure model: {model_name}")
            
            model_path = self._get_model_path(model_name)
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_path)
            model = TableTransformerForObjectDetection.from_pretrained(
                model_name,
                cache_dir=model_path,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)
            model.eval()
            
            self._models["table_structure"] = model
            self._processors["table_structure"] = processor
            logger.info("Table structure model loaded successfully")
            return model, processor
    
    def load_element_detection_model(self) -> tuple:
        """Load element detection model (LayoutLM for element classification)."""
        with self._lock:
            if "element_detection" in self._models:
                return self._models["element_detection"], self._processors["element_detection"]
            
            model_name = settings.element_detection_model
            logger.info(f"Loading element detection model: {model_name}")
            
            model_path = self._get_model_path(model_name)
            processor = AutoProcessor.from_pretrained(model_name, cache_dir=model_path)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name,
                cache_dir=model_path,
                torch_dtype=self._dtype,
                low_cpu_mem_usage=True,
            ).to(self._device)
            model.eval()
            
            self._models["element_detection"] = model
            self._processors["element_detection"] = processor
            logger.info("Element detection model loaded successfully")
            return model, processor
    
    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Preprocess image for model input."""
        # Resize if too large
        max_size = settings.max_image_size
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to RGB if needed
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        return image
    
    def classify_document(self, image: Image.Image, candidate_types: List[str]) -> Dict[str, Any]:
        """Classify document type using LayoutLM."""
        model, processor = self.load_classifier_model()
        image = self.preprocess_image(image)
        
        # For classification, we'd typically fine-tune LayoutLM on document types
        # This is a simplified implementation
        encoding = processor(
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=settings.max_sequence_length,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Map to candidate types (simplified)
        scores = probs[0].cpu().numpy()
        predicted_idx = scores.argmax()
        
        return {
            "predicted_type": candidate_types[predicted_idx] if predicted_idx < len(candidate_types) else "unknown",
            "confidence": float(scores[predicted_idx]),
            "all_scores": {t: float(s) for t, s in zip(candidate_types, scores[:len(candidate_types)])}
        }
    
    def extract_fields(self, image: Image.Image, field_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Extract fields from document using LayoutLM/Donut."""
        # Use Donut for general extraction
        model, processor = self.load_donut_model()
        image = self.preprocess_image(image)
        
        # Prepare prompt for Donut
        task_prompt = "<s_cord-v2>"
        decoder_input_ids = processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt"
        ).input_ids
        
        pixel_values = processor(image, return_tensors="pt").pixel_values.to(self._device)
        
        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids.to(self._device),
                max_length=settings.max_sequence_length,
                early_stopping=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                bad_words_ids=[[processor.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        sequence = processor.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
        sequence = sequence.replace(task_prompt, "").strip()
        
        # Parse JSON output from Donut
        import json
        try:
            extracted = json.loads(sequence)
        except json.JSONDecodeError:
            extracted = {}
        
        # Map to field schema
        fields = {}
        for field_key, field_def in field_schema.items():
            value = extracted.get(field_key, "")
            fields[field_key] = {
                "value": value,
                "confidence": 0.8,  # Donut doesn't provide per-field confidence
                "bbox": {"x": 0, "y": 0, "width": 0, "height": 0},
            }
        
        return {
            "fields": fields,
            "overall_confidence": 0.8,
            "model_version": settings.donut_model,
        }
    
    def detect_tables(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect tables in document using Table Transformer."""
        model, processor = self.load_table_detection_model()
        image = self.preprocess_image(image)
        
        encoding = processor(images=image, return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            outputs = model(**encoding)
        
        # Post-process detections
        target_sizes = torch.tensor([image.size[::-1]]).to(self._device)
        results = processor.post_process_object_detection(
            outputs, threshold=settings.confidence_threshold, target_sizes=target_sizes
        )[0]
        
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            tables.append({
                "bbox": {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
                "confidence": round(score.item(), 3),
            })
        
        return tables
    
    def detect_table_structure(self, image: Image.Image, table_bbox: Dict[str, float]) -> Dict[str, Any]:
        """Detect table structure (rows, columns, cells)."""
        model, processor = self.load_table_structure_model()
        
        # Crop table region
        x, y, w, h = table_bbox["x"], table_bbox["y"], table_bbox["width"], table_bbox["height"]
        table_image = image.crop((x, y, x + w, y + h))
        table_image = self.preprocess_image(table_image)
        
        encoding = processor(images=table_image, return_tensors="pt").to(self._device)
        
        with torch.no_grad():
            outputs = model(**encoding)
        
        target_sizes = torch.tensor([table_image.size[::-1]]).to(self._device)
        results = processor.post_process_object_detection(
            outputs, threshold=settings.confidence_threshold, target_sizes=target_sizes
        )[0]
        
        # Parse structure into rows/columns
        cells = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            cells.append({
                "bbox": {"x": box[0], "y": box[1], "width": box[2] - box[0], "height": box[3] - box[1]},
                "confidence": round(score.item(), 3),
                "label": label.item(),
            })
        
        return {"cells": cells}
    
    def detect_elements(self, image: Image.Image, element_types: List[str]) -> List[Dict[str, Any]]:
        """Detect document elements (checkboxes, signatures, etc.)."""
        model, processor = self.load_element_detection_model()
        image = self.preprocess_image(image)
        
        encoding = processor(
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=settings.max_sequence_length,
        ).to(self._device)
        
        with torch.no_grad():
            outputs = model(**encoding)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # Simplified: return detected elements with confidence
        elements = []
        # This would require a model trained for element detection
        # For now, return empty list
        return elements
    
    def get_loaded_models(self) -> Dict[str, str]:
        """Get info about loaded models."""
        return {name: str(type(model)) for name, model in self._models.items()}
    
    def health_check(self) -> Dict[str, Any]:
        """Health check for model manager."""
        return {
            "healthy": True,
            "device": self._device,
            "dtype": str(self._dtype),
            "loaded_models": list(self._models.keys()),
            "model_cache_dir": str(settings.model_cache_dir),
        }


# Global model manager instance
model_manager = ModelManager()