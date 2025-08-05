#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import torch
import re
from typing import Dict, Optional, List
from PIL import Image
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from transformers import LlavaForConditionalGeneration, LlavaProcessor

from ..model_clients import ModelClient, ConfigurationError, ImageProcessingError

logger = logging.getLogger("GUIExplorerClient")

class GUIExplorerClient(ModelClient):
    """Specialized client for GUIExplorer (LLaVA OneVision based) model"""
    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        super().__init__(model_name, api_key, config, use_component_detector)
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = config.get("weights_path", "")
        
        self._load_model()
    
    def _validate_model_specific_config(self) -> None:
        """Validate GUIExplorer specific configuration"""
        if not self.config:
            raise ConfigurationError("Configuration is required for offline models")
            
        weights_path = self.config.get("weights_path")
        if not weights_path:
            raise ConfigurationError("Model weights path is required")
    
    def _load_model(self) -> None:
        """Load GUIExplorer model with proper memory management"""
        try:
            # Clear GPU memory first
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            model_path = self.weights_path
            if not os.path.exists(model_path):
                model_path = os.path.join("models", self.weights_path)
            
            if not os.path.exists(model_path):
                raise ConfigurationError(f"Model path does not exist: {model_path}")
                
            logger.info(f"Loading GUIExplorer from: {model_path}")
            
            # Load processor with error handling
            try:
                self.processor = AutoProcessor.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                logger.info("Loaded AutoProcessor successfully")
            except Exception as e:
                logger.warning(f"AutoProcessor failed: {e}, trying LlavaProcessor")
                try:
                    self.processor = LlavaProcessor.from_pretrained(
                        model_path,
                        trust_remote_code=True
                    )
                except Exception as e2:
                    logger.error(f"Both processors failed: {e2}")
                    raise ConfigurationError(f"Failed to load processor: {e2}")
            
            # Fix processor configuration issues
            if hasattr(self.processor, 'image_processor'):
                if hasattr(self.processor.image_processor, 'patch_size'):
                    if self.processor.image_processor.patch_size is None:
                        self.processor.image_processor.patch_size = 14
                        logger.info("Fixed image_processor patch_size to 14")
                
                # Reduce image resolution to save memory
                if hasattr(self.processor.image_processor, 'size'):
                    if isinstance(self.processor.image_processor.size, dict):
                        self.processor.image_processor.size = {"height": 672, "width": 672}
                    else:
                        self.processor.image_processor.size = 672
                    logger.info("Set image processor size to 672x672 for memory optimization")
            
            # Load tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                # Ensure pad token is set
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                    logger.info("Set pad_token to eos_token")
            except Exception as e:
                logger.warning(f"Failed to load tokenizer: {e}")
            
            # Load model with optimized settings
            try:
                # First try LlavaForConditionalGeneration
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    model_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                    max_memory={i: "6GB" for i in range(torch.cuda.device_count())}
                )
                logger.info("Loaded LlavaForConditionalGeneration successfully")
            except Exception as e:
                logger.warning(f"LlavaForConditionalGeneration failed: {e}, trying AutoModel")
                try:
                    self.model = AutoModel.from_pretrained(
                        model_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={i: "6GB" for i in range(torch.cuda.device_count())}
                    )
                    logger.info("Loaded AutoModel successfully")
                except Exception as e2:
                    logger.error(f"Both model loading methods failed: {e2}")
                    raise ConfigurationError(f"Failed to load model: {e2}")
            
            # Ensure model is in eval mode
            if self.model:
                self.model.eval()
                logger.info("GUIExplorer model loaded successfully and set to eval mode")
            
        except Exception as e:
            logger.error(f"Error loading GUIExplorer: {str(e)}")
            raise ConfigurationError(f"Failed to load GUIExplorer: {str(e)}")
    
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """Make prediction with GUIExplorer"""
        try:
            # Validate image path
            if not os.path.exists(image_path):
                alt_path = os.path.join('element_detection', os.path.basename(image_path))
                if os.path.exists(alt_path):
                    image_path = alt_path
                else:
                    return {"error": f"Image not found: {image_path}", "raw_response": ""}
            
            # Load and process image
            try:
                image = Image.open(image_path).convert('RGB')
                # Resize image to reduce memory usage
                max_size = 1024
                if image.width > max_size or image.height > max_size:
                    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                    logger.info(f"Resized image from {image_path} to {image.size}")
            except Exception as e:
                return {"error": f"Failed to load image: {str(e)}", "raw_response": ""}
            
            # Format prompt for LLaVA
            formatted_prompt = self._format_prompt(prompt)
            
            # Process inputs
            try:
                # Check if processor has proper chat template
                if hasattr(self.processor, 'apply_chat_template') and hasattr(self.processor, 'chat_template') and self.processor.chat_template is not None:
                    # Use chat template if available
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": formatted_prompt}
                            ]
                        }
                    ]
                    prompt_text = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
                    inputs = self.processor(images=image, text=prompt_text, return_tensors="pt")
                else:
                    # Fallback to direct processing
                    inputs = self.processor(images=image, text=formatted_prompt, return_tensors="pt")
                
                # Move inputs to device
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                
            except Exception as e:
                logger.error(f"Error processing inputs: {str(e)}")
                return {"error": f"Input processing failed: {str(e)}", "raw_response": ""}
            
            # Generate response
            try:
                with torch.no_grad():
                    # Clear cache before generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Generate with conservative settings
                    generation_config = {
                        "max_new_tokens": 256,
                        "do_sample": False,
                        "temperature": 0.1,
                        "pad_token_id": self.processor.tokenizer.eos_token_id if self.processor.tokenizer else None,
                        "use_cache": False,  # Disable cache to save memory
                    }
                    
                    # Remove None values
                    generation_config = {k: v for k, v in generation_config.items() if v is not None}
                    
                    outputs = self.model.generate(**inputs, **generation_config)
                    
                    # Clear cache after generation
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Error during generation: {str(e)}")
                return {"error": f"Generation failed: {str(e)}", "raw_response": ""}
            
            # Decode response
            try:
                if 'input_ids' in inputs:
                    response = self.processor.decode(
                        outputs[0][inputs['input_ids'].shape[1]:], 
                        skip_special_tokens=True
                    )
                else:
                    response = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                response = response.strip()
                
            except Exception as e:
                logger.error(f"Error decoding response: {str(e)}")
                return {"error": f"Decoding failed: {str(e)}", "raw_response": ""}
            
            # Extract coordinates from response
            coordinates = self._extract_coordinates(response)
            
            result = {"raw_response": response, "error": None}
            if coordinates:
                result["coordinates"] = coordinates
                
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}", "raw_response": ""}
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for LLaVA model"""
        # Add system prompt if needed
        if not prompt.startswith("You are"):
            system_prompt = "You are a helpful assistant that can understand and interact with GUI elements. "
            prompt = system_prompt + prompt
        
        return prompt
    
    def _extract_coordinates(self, response: str) -> Optional[List[int]]:
        """Extract coordinates from response"""
        if not response:
            return None
        
        # Pattern for [x, y] format
        pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        match = re.search(pattern, response)
        
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            
            # Convert to pixel coordinates if normalized
            if x <= 1.0 and y <= 1.0:
                # Assume original image size (can be adjusted)
                x, y = int(x * 1920), int(y * 1080)
            
            return [int(x), int(y)]
        
        # Pattern for (x, y) format
        pattern = r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)'
        match = re.search(pattern, response)
        
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            if x <= 1.0 and y <= 1.0:
                x, y = int(x * 1920), int(y * 1080)
            return [int(x), int(y)]
        
        # Pattern for "x, y" format
        pattern = r'(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)'
        match = re.search(pattern, response)
        
        if match:
            x, y = float(match.group(1)), float(match.group(2))
            if x <= 1.0 and y <= 1.0:
                x, y = int(x * 1920), int(y * 1080)
            return [int(x), int(y)]
        
        return None
    
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if self.model:
                del self.model
                self.model = None
            if self.processor:
                del self.processor
                self.processor = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
        super().cleanup()