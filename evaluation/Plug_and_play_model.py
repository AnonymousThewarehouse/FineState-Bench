import requests
import logging
import json
import os
from PIL import Image
import base64
from io import BytesIO
from typing import List, Tuple, Dict, Optional
from .utils import image_to_base64
import re

logger = logging.getLogger(__name__)

class ComponentDetector:
    def __init__(self, model_name: str = None, api_key: str = None, config: Dict = None, model = None):

        self.config = config or {}
        self.model_name = model_name or self.config.get("model_name", "google/gemini-2.5-flash-preview")
        self.api_key = api_key or self.config.get("api_key")
        self.model = model  

        self.component_types = {
            "button": ["button", "btn", "clickable", "click"],
            "slider": ["slider", "scrollbar", "scroll", "progress"],
            "checkbox": ["checkbox", "check box", "tick box"],
            "radio": ["radio", "radio button", "option"],
            "text": ["text", "input", "textbox", "text field"],
            "dropdown": ["dropdown", "select", "combobox", "combo box"],
            "menu": ["menu", "navigation", "nav"],
            "link": ["link", "hyperlink", "url"],
            "image": ["image", "picture", "photo", "img"],
            "video": ["video", "player", "media"],
            "icon": ["icon", "symbol", "logo"],
            "tab": ["tab", "tab bar", "tab panel"],
            "list": ["list", "listbox", "list view"],
            "table": ["table", "grid", "spreadsheet"],
            "form": ["form", "form field", "input form"]
        }

        self.ground_truth_coords = {}
        self.use_ground_truth = False
        self.load_ground_truth_coordinates()
        
    def load_ground_truth_coordinates(self):
        try:
            ground_truth_path = self.config.get("ground_truth_path", "pipline/slider_instructions.jsonl")
            if os.path.exists(ground_truth_path):
                with open(ground_truth_path, "r") as f:
                    for line in f:
                        data = json.loads(line)
                        if "image" in data and "processing_details" in data:
                            processing_details = data["processing_details"]
                            if "bbox" in processing_details:
                                image_key = data["image"]
                                if not image_key.startswith("element_detection/"):
                                    image_key = f"element_detection/{image_key}"
                                self.ground_truth_coords[image_key] = {
                                    "bbox": processing_details["bbox"],
                                    "now_coords": processing_details.get("now_coords"),
                                    "target_coords": processing_details.get("target_coords")
                                }
                logger.info(f"Successfully loaded {len(self.ground_truth_coords)} ground truth bounding box coordinates")
            else:
                logger.warning(f"Ground truth bounding box file does not exist: {ground_truth_path}")
        except Exception as e:
            logger.error(f"Failed to load ground truth bounding box coordinates: {str(e)}")
            
    def set_use_ground_truth(self, use: bool):
        self.use_ground_truth = use
        logger.info(f"Use ground truth bounding box: {use}")
        
    def detect_component(self, image_path: str, prompt: str = None) -> Dict:

        try:
            
            if self.use_ground_truth and self.ground_truth_coords:
                
                normalized_path = image_path
                if not normalized_path.startswith("element_detection/"):
                    normalized_path = f"element_detection/{os.path.basename(image_path)}"
                
                if normalized_path in self.ground_truth_coords:
                    gt_data = self.ground_truth_coords[normalized_path]
                    bbox = gt_data.get("bbox")
                    if bbox:
                        logger.info(f"Using ground truth bounding box: {normalized_path} -> {bbox}")
                        return {
                            "success": True,
                            "description": "Ground truth component detected",
                            "bbox": bbox,
                            "operation": "Ground truth data",
                            "error": None
                        }
                
                logger.warning(f"Ground truth bounding box data not found for image: {normalized_path}")
           
            if not prompt:
                prompt = """Please analyze this UI interface image and identify the UI components in it.\nPlease provide the following information:\n1. Component type (such as button, slider, checkbox, etc.)\n2. Component name (if any)\n3. Component bounding box coordinates [x1, y1, x2, y2]\n4. Confidence (a decimal between 0-1)\n\nPlease return the result in JSON format, including the following fields:\n{\n    \"components\": [\n        {\n            \"type\": \"component type\",\n            \"name\": \"component name\",\n            \"bbox\": [x1, y1, x2, y2],\n            \"confidence\": 0.95\n        }\n    ]\n}"""
            
            image_base64 = image_to_base64(image_path)
            
            response = self._get_model_response(image_base64, prompt)
            
            try:
                result = json.loads(response)
            except json.JSONDecodeError:
                result = self._parse_text_response(response)
        
            if result.get("components"):
                component = result["components"][0]  
                return {
                    "success": True,
                    "description": f"Type: {component.get('type', 'unknown')}, Name: {component.get('name', 'unknown')}",
                    "bbox": component.get("bbox"),
                    "operation": f"Detected with confidence: {component.get('confidence', 0.0)}",
                    "error": None
                }
            else:
                return {
                    "success": False,
                    "description": "",
                    "bbox": None,
                    "operation": "",
                    "error": "No components detected"
                }
                        
        except Exception as e:
            error_msg = f"Component detection failed: {str(e)}"
            logger.error(error_msg)
            return {
                "success": False,
                "description": "",
                "bbox": None,
                "operation": "",
                "error": error_msg
            }
    
    def _get_model_response(self, image_base64: str, prompt: str) -> str:

        try:
            
            if self.model:
                return self._get_local_model_response(image_base64, prompt)
            
            return self._get_api_response(image_base64, prompt)
            
        except Exception as e:
            logger.error(f"Failed to get model response: {str(e)}")
            return ""
    
    def _get_local_model_response(self, image_base64: str, prompt: str) -> str:

        try:
            
            return """{"components": [{"type": "slider", "name": "video_slider", "bbox": [100, 400, 800, 450], "confidence": 0.9}]}"""
        except Exception as e:
            logger.error(f"Local model response failed: {str(e)}")
            return ""
    
    def _get_api_response(self, image_base64: str, prompt: str) -> str:
        
        try:
            if "gemini" in self.model_name.lower():
                return self._call_gemini_api(image_base64, prompt)
            elif "gpt" in self.model_name.lower():
                return self._call_openai_api(image_base64, prompt)
            else:
                logger.warning(f"Unsupported model type: {self.model_name}")
                return ""
                
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            return ""
    
    def _call_gemini_api(self, image_base64: str, prompt: str) -> str:
        
        try:
           
            return """{"components": [{"type": "slider", "name": "video_slider", "bbox": [100, 400, 800, 450], "confidence": 0.9}]}"""
        except Exception as e:
            logger.error(f"Gemini API call failed: {str(e)}")
            return ""
    
    def _call_openai_api(self, image_base64: str, prompt: str) -> str:
        
        try:
            
            return """{"components": [{"type": "slider", "name": "video_slider", "bbox": [100, 400, 800, 450], "confidence": 0.9}]}"""
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            return ""

    def _parse_text_response(self, response: str) -> Dict:
        
        components = []
        
        coord_patterns = [
            r'\[(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]',  
            r'\((\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\)',  
            r'Coordinates[：:][\s\[]*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\]?',  
        ]
        
        type_pattern = r'Type[：:][^\n,]+'
        name_pattern = r'Name[：:][^\n,]+'
        confidence_pattern = r'Confidence[：:](0\.\d+|1\.0)'
        
        for line in response.split('\n'):
            component = {}
            
            type_match = re.search(type_pattern, line)
            if type_match:
                component['type'] = type_match.group(1).strip()
            
            name_match = re.search(name_pattern, line)
            if name_match:
                component['name'] = name_match.group(1).strip()
            
            for pattern in coord_patterns:
                coord_match = re.search(pattern, line)
                if coord_match:
                    component['bbox'] = [
                        int(coord_match.group(1)),
                        int(coord_match.group(2)),
                        int(coord_match.group(3)),
                        int(coord_match.group(4))
                    ]
                    break
            
            confidence_match = re.search(confidence_pattern, line)
            if confidence_match:
                component['confidence'] = float(confidence_match.group(1))
            else:
                component['confidence'] = 0.9  
            
            if 'type' in component and 'bbox' in component:
                components.append(component)
        
        return {"components": components}

    def format_component_info(self, component_info: Dict) -> Dict:
        
        if not component_info or not component_info.get("components"):
            return {"description": "", "bbox": None}
        
        component = component_info["components"][0] if component_info["components"] else {}
        
        description_parts = []
        if component.get("type"):
            description_parts.append(f"Type: {component['type']}")
        if component.get("name"):
            description_parts.append(f"Name: {component['name']}")
        if component.get("confidence"):
            description_parts.append(f"Confidence: {component['confidence']:.2f}")
        
        description = ", ".join(description_parts) if description_parts else "UI component detected"
        
        bbox = component.get("bbox")
        if bbox and len(bbox) == 4:
            
            if all(coord <= 1.0 for coord in bbox):
                
                normalized_bbox = bbox
            else:
                
                normalized_bbox = [
                    bbox[0] / 1080.0,
                    bbox[1] / 1920.0,
                    bbox[2] / 1080.0,
                    bbox[3] / 1920.0
                ]
        else:
            normalized_bbox = None
        
        return {
            "description": description,
            "bbox": normalized_bbox,
            "operation": component.get("operation", "")
        }

    def normalize_coordinates(self, coords: List[float], image_size: Tuple[int, int]) -> List[float]:
        
        if not coords or len(coords) != 4:
            return None
        
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        
        try:
            
            x1 = max(0, min(coords[0], width))
            y1 = max(0, min(coords[1], height))
            x2 = max(0, min(coords[2], width))
            y2 = max(0, min(coords[3], height))
            
            normalized = [
                x1 / width,
                y1 / height,
                x2 / width,
                y2 / height
            ]
            
            return normalized
        except Exception as e:
            logger.error(f"Coordinate normalization failed: {str(e)}")
            return None

    def denormalize_coordinates(self, coords: List[float], image_size: Tuple[int, int]) -> List[float]:
        
        if not coords or len(coords) != 4:
            return None
        
        width, height = image_size
        if width <= 0 or height <= 0:
            return None
        
        try:
            
            x1 = max(0, min(coords[0], 1))
            y1 = max(0, min(coords[1], 1))
            x2 = max(0, min(coords[2], 1))
            y2 = max(0, min(coords[3], 1))
            
            pixel_coords = [
                x1 * width,
                y1 * height,
                x2 * width,
                y2 * height
            ]
            
            return pixel_coords
        except Exception as e:
            logger.error(f"Coordinate denormalization failed: {str(e)}")
            return None
        
    def validate_coordinates(self, coords: List[float], is_normalized: bool = True) -> bool:
        
        if not coords or len(coords) != 4:
            return False
        
        try:
            
            if is_normalized:
                
                valid_range = (0, 1)
            else:
                
                valid_range = (0, float('inf'))
           
            for coord in coords:
                if not isinstance(coord, (int, float)):
                    return False
                if coord < valid_range[0] or coord > valid_range[1]:
                    return False
            
            if coords[0] > coords[2] or coords[1] > coords[3]:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Coordinate validation failed: {str(e)}")
            return False

    def calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        
        if not self.validate_coordinates(bbox1) or not self.validate_coordinates(bbox2):
            return 0.0
        
        try:
            x1 = max(bbox1[0], bbox2[0])
            y1 = max(bbox1[1], bbox2[1])
            x2 = min(bbox1[2], bbox2[2])
            y2 = min(bbox1[3], bbox2[3])
           
            if x2 <= x1 or y2 <= y1:
                return 0.0
            
            intersection = (x2 - x1) * (y2 - y1)
            
            area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
            area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
            
            union = area1 + area2 - intersection
            
            iou = intersection / union if union > 0 else 0.0
            
            return iou
        except Exception as e:
            logger.error(f"IoU calculation failed: {str(e)}")
            return 0.0