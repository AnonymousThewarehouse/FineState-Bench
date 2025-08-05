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
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
    # For PEFT/LoRA
    try:
        from peft import PeftModel, PeftConfig
        PEFT_AVAILABLE = True
    except ImportError:
        PEFT_AVAILABLE = False
        # We'll log this warning later when the logger is initialized
except ImportError:
    raise ImportError("Transformers library not found. Please install with 'pip install transformers'")

# Avoid circular import
from ..model_clients import ModelClient, ConfigurationError, ImageProcessingError

logger = logging.getLogger("LoRAAdapterClient")

class LoRAAdapterClient(ModelClient):
    """Specialized client for LoRA adapter models like web-llama2-13b-adapter"""
    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        """
        Initialize LoRA adapter client
        
        Args:
            model_name: Name of the model
            api_key: Optional API key (not used for offline models)
            config: Configuration dictionary
            use_component_detector: Whether to use component detection
        """
        # Log initialization
        logger.info(f"Initializing LoRA adapter client for model: {model_name}")
        
        # Set default model architecture
        if config is None:
            config = {}
            
        # Set LoRA specific default configs
        if "llm_config" not in config:
            config["llm_config"] = {
                "architectures": ["LlamaForCausalLM"],
                "model_type": "llama",
                "trust_remote_code": True
            }
        
        # Call parent constructor
        super().__init__(model_name, api_key, config, use_component_detector)
        
        # Initialize model attributes
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = config.get("weights_path", "")
        self.base_model_path = config.get("base_model_path", "meta-llama/Llama-2-13b-hf")
        
        # Load model components
        self._load_model()
    
    def _validate_model_specific_config(self) -> None:
        """Validate LoRA adapter specific configuration"""
        if not self.config:
            raise ConfigurationError("Configuration is required for offline models")
            
        weights_path = self.config.get("weights_path")
        if not weights_path:
            raise ConfigurationError("Model weights path is required")
            
        # Additional validation for LoRA adapters
        if not PEFT_AVAILABLE:
            raise ConfigurationError("PEFT library is required for LoRA adapters, please install with 'pip install peft'")
    
    def _find_model_path(self) -> str:
        """Find the model path in various locations"""
        # Try direct path first
        if os.path.exists(self.weights_path):
            logger.info(f"Using local model path: {self.weights_path}")
            return self.weights_path
            
        # Try models directory
        local_path = os.path.join("models", self.weights_path)
        if os.path.exists(local_path):
            logger.info(f"Found cached model path: {local_path}")
            return local_path
            
        # Try to find snapshot directory (web-llama2-13b-adapter)
        if "web-llama2" in self.weights_path.lower():
            possible_snapshot_bases = [
                os.path.join("models", "web-llama2-13b-adapter", "models--xhan77--web-llama2-13b-adapter", "snapshots"),
                os.path.join("models", self.weights_path, "models--xhan77--web-llama2-13b-adapter", "snapshots")
            ]
            
            for snapshot_base in possible_snapshot_bases:
                if os.path.exists(snapshot_base):
                    # Find the first subdirectory in the snapshot directory
                    for snapshot_dir in os.listdir(snapshot_base):
                        snapshot_path = os.path.join(snapshot_base, snapshot_dir)
                        if os.path.isdir(snapshot_path) and os.path.exists(os.path.join(snapshot_path, "adapter_config.json")):
                            logger.info(f"Found model snapshot path: {snapshot_path}")
                            return snapshot_path
            
        raise ConfigurationError(f"Model files not found: {self.weights_path}")
    
    def _load_model(self) -> None:
        """Load the LoRA adapter model"""
        try:
            if not hasattr(self, 'model_path') or not self.model_path:
                self.model_path = self._find_model_path()
            
            logger.info(f"Loading LoRA adapter model from: {self.model_path}")
            
            # Validate model path
            if not os.path.exists(self.model_path):
                raise ConfigurationError(f"Model path does not exist: {self.model_path}")
                
            # Check adapter_config.json file
            config_path = os.path.join(self.model_path, "adapter_config.json")
            if not os.path.exists(config_path):
                raise ConfigurationError(f"Adapter config file not found at {config_path}")
                
            # Read config file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                logger.info(f"Base model name: {config_data.get('base_model_name_or_path', 'unknown')}")
                logger.info(f"PEFT type: {config_data.get('peft_type', 'unknown')}")
                logger.info(f"Task type: {config_data.get('task_type', 'unknown')}")
            
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
            
            # Load base model first - handle offline mode
            logger.info(f"Loading base model: {self.base_model_path}")
            
            # Check if base model is available locally
            base_model_local_path = None
            if os.path.exists(self.base_model_path):
                base_model_local_path = self.base_model_path
            else:
                # Try to find local base model in models directory
                potential_local_paths = [
                    os.path.join("models", os.path.basename(self.base_model_path)),
                    os.path.join("models", self.base_model_path.replace("/", "_")),
                    os.path.join("models", "Llama-2-13b-hf"),
                    os.path.join("models", "llama-2-13b-hf")
                ]
                
                for path in potential_local_paths:
                    if os.path.exists(path):
                        base_model_local_path = path
                        logger.info(f"Found local base model at: {path}")
                        break
            
            if base_model_local_path:
                try:
                    base_model = AutoModelForCausalLM.from_pretrained(
                        base_model_local_path,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                        local_files_only=True  # Force offline mode
                    )
                    logger.info("Successfully loaded base model from local path")
                except Exception as local_error:
                    logger.error(f"Failed to load local base model: {local_error}")
                    raise ConfigurationError(f"Base model not available locally: {self.base_model_path}")
            else:
                # Base model not available locally, this is a configuration issue
                logger.error(f"Base model {self.base_model_path} not available locally")
                logger.error("LoRA adapter requires base model to function properly")
                logger.error("This model requires external dependencies that are not available")
                raise ConfigurationError(f"Base model not available locally: {self.base_model_path}. LoRA adapters require their specific base model to function properly. Please download the base model or use a different model.")
            
            # Load tokenizer from base model
            logger.info("Loading tokenizer from base model...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                base_model_local_path if base_model_local_path else self.base_model_path,
                trust_remote_code=True,
                local_files_only=True if base_model_local_path else False
            )
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                logger.info("Added padding token")
            
            # Load LoRA adapter
            logger.info("Loading LoRA adapter...")
            self.model = PeftModel.from_pretrained(
                base_model,
                self.model_path,
                torch_dtype=dtype
            )
            
            logger.info("LoRA adapter model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading LoRA adapter model: {str(e)}")
            raise ConfigurationError(f"Failed to load LoRA adapter model: {str(e)}")
    
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """
        Make a prediction with LoRA adapter model
        
        Args:
            prompt: Text prompt
            image_path: Path to image file (not used for text-only model)
            image_base64: Optional pre-encoded image (not used for text-only model)
            
        Returns:
            Dict: Prediction results with raw_response
        """
        try:
            # Verify model is loaded
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Model not loaded")
                return {"error": "Model not loaded", "raw_response": ""}
            
            # Prepare prompt (image_path is ignored for text-only model)
            enhanced_prompt = self.prepare_prompt(prompt, image_path)
            
            # Use LoRA adapter for text generation
            logger.info("Generating response with LoRA adapter")
            
            # Format the prompt for GUI interaction
            gui_prompt = f"""Please analyze the UI element and provide coordinates for the requested action.

Task: {enhanced_prompt}

Response format:
1. Component Description: [Brief description of the target UI element]
2. Interaction Coordinates: [x, y] (normalized coordinates between 0 and 1)
3. Reasoning: [Explain why you chose this location]

Please provide exact coordinates in [x, y] format where x and y are decimal numbers between 0 and 1."""
            
            # Tokenize input
            inputs = self.tokenizer(
                gui_prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            # Move to device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove input from response
            if gui_prompt in response:
                response = response.split(gui_prompt)[-1].strip()
            
            logger.info(f"Generated response: {response[:100]}...")
            return {"raw_response": response, "error": None}
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}", "raw_response": ""}
            
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        super().cleanup()