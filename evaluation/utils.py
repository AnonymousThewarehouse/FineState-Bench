import os
import base64
import logging
from typing import Optional, Dict, List
from PIL import Image
import json
import yaml

logger = logging.getLogger(__name__)

def image_to_base64(image_path: str) -> Optional[str]:
    
    try:
        
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None
            
        
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        base64_str = base64.b64encode(image_data).decode("utf-8")
        return base64_str
        
    except Exception as e:
        logger.error(f"Image to Base64 conversion failed: {str(e)}")
        return None

def resize_image_if_needed(image_path: str, max_size: int = 1024 * 1024) -> Optional[str]:
    
    try:
        
        if not os.path.exists(image_path):
            logger.error(f"Image file does not exist: {image_path}")
            return None
            
        
        file_size = os.path.getsize(image_path)
        
        if file_size <= max_size:
            return image_path
        
        with Image.open(image_path) as img:
            
            ratio = (max_size / file_size) ** 0.5
            
            new_width = int(img.width * ratio)
            new_height = int(img.height * ratio)
            
            resized_img = img.resize((new_width, new_height), Image.LANCZOS)

            filename, ext = os.path.splitext(image_path)
            new_path = f"{filename}_resized{ext}"    
            
            resized_img.save(new_path, quality=85)            
            logger.info(f"Image resized: {image_path} -> {new_path}")
            return new_path            
    except Exception as e:
        logger.error(f"Image resizing failed: {str(e)}")
        return image_path  

def load_slider_instructions(file_path: str, element_detection_dir: str = "element_detection") -> tuple[list, bool]:

    converted_data = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        
                        if "image" in item and not os.path.isabs(item["image"]):
                            item["image"] = os.path.join(element_detection_dir, item["image"])
                        converted_data.append(item)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing JSON line: {str(e)}, line content: {line[:50]}...")        

        missing_images = []
        fixed_count = 0        
        for i, item in enumerate(converted_data):
            image_path = item.get("image")
            if not image_path:
                continue                
            if os.path.exists(image_path):
                continue                
            basename = os.path.basename(image_path)
            alt_path = os.path.join(element_detection_dir, basename)            
            if os.path.exists(alt_path):
                converted_data[i]["image"] = alt_path
                fixed_count += 1
            else:
                missing_images.append(image_path)        
        logger.info(f"Image validation: fixed {fixed_count} image paths")
        if missing_images:
            logger.warning(f"Cannot find {len(missing_images)} images: {missing_images[:5]}{'...' if len(missing_images) > 5 else ''}")        
        return converted_data, len(missing_images) == 0        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        return [], False

class ConfigLoader:
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger("ConfigLoader")
        self.config = self._load_config()
    
    def _load_config(self) -> Dict:
        try:
            if not os.path.exists(self.config_path):
                self.logger.error(f"Configuration file does not exist: {self.config_path}")
                return {}
                
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if "PredictionModel" in config or "PluggableDetectionModel" in config:
                models = {}

                for model in config.get("PredictionModel", []):
                    name = model.get("name")
                    if name:
                        models[name] = {
                            "model_type": model.get("type", "online"),
                            "api_key": model.get("api_key", ""),
                            "api_base": model.get("api_url", "https://openrouter.ai/api/v1"),
                            "extra_headers": model.get("extra_headers", {}),
                            "params": model.get("params", {}),
                            "weights_path": model.get("weights_path", ""),
                            "description": model.get("description", "")
                        }

                for model in config.get("PluggableDetectionModel", []):
                    name = model.get("name")
                    if name:
                        models[name] = {
                            "model_type": model.get("type", "online"),
                            "api_key": model.get("api_key", ""),
                            "api_base": model.get("api_url", "https://openrouter.ai/api/v1"),
                            "extra_headers": model.get("extra_headers", {}),
                            "params": model.get("params", {})
                        }
                        
                return {"models": models}
            else:
                models = {}
                for model_name, model_config in config.items():
                    if isinstance(model_config, dict):
                        standardized_config = {
                            "model_type": model_config.get("model_type", "online"),
                            "api_key": model_config.get("api_key", ""),
                            "api_url": model_config.get("api_url", ""),
                            "api_base": model_config.get("api_url", "https://openrouter.ai/api/v1"),
                            "extra_headers": model_config.get("extra_headers", {}),
                            "params": model_config.get("params", {}),
                            "weights_path": model_config.get("weights_path", ""),
                            "description": model_config.get("description", "")
                        }
                        models[model_name] = standardized_config
                
                self.logger.info(f"Successfully loaded configuration file: {self.config_path}, found {len(models)} models")
                return {"models": models}
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {str(e)}")
            return {"models": {}}
    
    def get_models(self) -> List[str]:
        if not self.config or "models" not in self.config:
            return []
        return list(self.config["models"].keys())
    
    def get_model_config(self, model_name: str) -> Optional[Dict]:
        if not self.config or "models" not in self.config:
            self.logger.warning(f"Model configuration not found: {model_name}")
            return None
            
        model_config = self.config["models"].get(model_name)
        if not model_config:
            self.logger.warning(f"Model not found: {model_name}")
            return None
            
        return model_config
    
    def get_api_key(self, model_name: str) -> Optional[str]:
        model_config = self.get_model_config(model_name)
        if model_config:
            return model_config.get("api_key")
        return None