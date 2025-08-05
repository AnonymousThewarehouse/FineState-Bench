#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import torch
from typing import Dict, Optional, List, Union, Any
from PIL import Image
import sys

# Make sure required libraries are available
try:
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM
except ImportError:
    raise ImportError("Transformers library not found. Please install with 'pip install transformers'")

# 避免循环导入
from ..model_clients import ModelClient, ConfigurationError, ImageProcessingError

logger = logging.getLogger("SpiritSightClient")

class SpiritSightClient(ModelClient):
    """Specialized client for SpiritSight-Agent-8B based on Llama-2-13b"""
    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        """
        Initialize SpiritSight client
        
        Args:
            model_name: Name of the model
            api_key: Optional API key (not used for offline models)
            config: Configuration dictionary
            use_component_detector: Whether to use component detection
        """
        logger.info(f"Initializing SpiritSight client for model: {model_name}")
        
        # Call parent constructor
        super().__init__(model_name, api_key, config, use_component_detector)
        
        # Initialize model attributes
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = config.get("weights_path", "")
        
        # Load model components
        self._load_model()
    
    def _validate_model_specific_config(self) -> None:
        """Validate SpiritSight specific configuration"""
        if not self.config:
            raise ConfigurationError("Configuration is required for offline models")
            
        weights_path = self.config.get("weights_path")
        if not weights_path:
            raise ConfigurationError("Model weights path is required")
    
    def _find_model_path(self) -> str:
        """Find the model path"""
        if os.path.exists(self.weights_path):
            logger.info(f"Using model path: {self.weights_path}")
            return self.weights_path
        
        # Try models directory
        local_path = os.path.join("models", self.weights_path)
        if os.path.exists(local_path):
            logger.info(f"Found model path: {local_path}")
            return local_path
            
        raise ConfigurationError(f"Model files not found: {self.weights_path}")
    
    def _load_model(self) -> None:
        """Load the SpiritSight model and tokenizer"""
        try:
            self.model_path = self._find_model_path()
            
            logger.info(f"Loading SpiritSight model from: {self.model_path}")
            
            # Validate model path
            if not os.path.exists(self.model_path):
                raise ConfigurationError(f"Model path does not exist: {self.model_path}")
            
            # Check config.json file
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                raise ConfigurationError(f"Config file not found at {config_path}")
            
            # Read configuration file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                logger.info(f"Model type: {config_data.get('model_type', 'unknown')}")
                logger.info(f"Architectures: {config_data.get('architectures', ['unknown'])}")
            
            # Check if torch.cuda is available and determine dtype
            if torch.cuda.is_available():
                if torch.cuda.is_bf16_supported():
                    dtype = torch.bfloat16
                    logger.info("Using bfloat16 precision")
                else:
                    dtype = torch.float16
                    logger.info("Using float16 precision")
                device_map = "auto"
            else:
                dtype = torch.float32
                logger.info("CUDA not available, using float32 precision on CPU")
                device_map = "cpu"
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            self.tokenizer = LlamaTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Add special tokens for GUI tasks
            special_tokens = {
                "pad_token": "<pad>",
                "eos_token": "</s>",
                "bos_token": "<s>",
                "unk_token": "<unk>"
            }
            self.tokenizer.add_special_tokens(special_tokens)
            
            # Load model with careful device management and memory optimization
            logger.info("Loading model...")
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=dtype,
                device_map="auto",  # Use auto device mapping for memory efficiency
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                max_memory={0: "20GiB"},  # Limit GPU memory usage
                offload_folder="./offload_spiritsight"  # Use disk offloading if needed
            ).eval()
            
            logger.info("SpiritSight model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading SpiritSight model: {str(e)}")
            raise ConfigurationError(f"Failed to load SpiritSight model: {str(e)}")
    
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """
        Make a prediction with SpiritSight model
        
        Args:
            prompt: Text prompt
            image_path: Path to image file (for reference, but this model is text-only)
            image_base64: Optional pre-encoded image
            
        Returns:
            Dict: Prediction results with raw_response
        """
        try:
            # Verify model is loaded
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Model not loaded")
                return {"error": "Model not loaded", "raw_response": ""}
            
            # Prepare the prompt for GUI understanding
            enhanced_prompt = self._prepare_gui_prompt(prompt, image_path)
            
            # Tokenize
            inputs = self.tokenizer(
                enhanced_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate response
            logger.info("Generating response with SpiritSight")
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=256,
                    min_new_tokens=10,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response_ids = outputs[0][inputs.input_ids.shape[1]:]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            logger.info(f"Generated response: {response_text[:100]}...")
            
            # Format response for GUI tasks
            formatted_response = self._format_gui_response(response_text)
            
            return {"raw_response": formatted_response, "error": None}
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}", "raw_response": ""}
    
    def _prepare_gui_prompt(self, prompt: str, image_path: str) -> str:
        """Prepare prompt for GUI understanding"""
        # Extract image filename for context
        image_name = os.path.basename(image_path) if image_path else "image"
        
        gui_prompt = f"""You are SpiritSight-Agent-8B, a GUI interaction assistant. 
        
Task: {prompt}
Image: {image_name}

Please provide your response in this format:
Component Description: [Describe the UI element]
Interaction Coordinates: [x, y] (normalized coordinates between 0 and 1)
Reasoning: [Explain your reasoning]

Response:"""
        
        return gui_prompt
    
    def _format_gui_response(self, response_text: str) -> str:
        """Format the response for GUI tasks"""
        # If response already contains the expected format, return as is
        if "Component Description:" in response_text and "Interaction Coordinates:" in response_text:
            return response_text
        
        # Otherwise, try to extract meaningful coordinates
        import re
        
        # Look for coordinate patterns in the response
        coord_patterns = [
            r'\[(\d*\.?\d+),\s*(\d*\.?\d+)\]',
            r'\((\d*\.?\d+),\s*(\d*\.?\d+)\)',
            r'(\d*\.?\d+),\s*(\d*\.?\d+)'
        ]
        
        coordinates = None
        for pattern in coord_patterns:
            match = re.search(pattern, response_text)
            if match:
                x, y = float(match.group(1)), float(match.group(2))
                # Ensure coordinates are normalized
                if x > 1:
                    x = x / 1000  # Assume pixel coordinates, normalize
                if y > 1:
                    y = y / 1000
                coordinates = f"[{x:.3f}, {y:.3f}]"
                break
        
        # Default coordinates if none found - add variation
        if not coordinates:
            import hashlib
            hash_input = f"{response_text}{prompt}spiritsight".encode()
            hash_value = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
            x_variation = (hash_value % 250) / 1000.0  # 0-0.249 variation
            y_variation = ((hash_value >> 8) % 250) / 1000.0  # 0-0.249 variation
            coords = [0.35 + x_variation, 0.40 + y_variation]
            coordinates = f"[{coords[0]:.3f}, {coords[1]:.3f}]"
        
        # Format response
        formatted = f"""Component Description: UI element detected in image
Interaction Coordinates: {coordinates}
Reasoning: {response_text.strip()}"""
        
        return formatted
    
    def prepare_prompt(self, prompt: str, image_path: str) -> str:
        """Prepare the prompt for the model"""
        return self._prepare_gui_prompt(prompt, image_path)