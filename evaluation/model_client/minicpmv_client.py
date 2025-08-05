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
    from transformers import AutoProcessor, AutoTokenizer
    # MiniCPMV specific model
    try:
        from transformers import MiniCPMVForCausalLM
        MINICPMV_AVAILABLE = True
    except ImportError:
        MINICPMV_AVAILABLE = False
        # We'll log this warning later when the logger is initialized
except ImportError:
    raise ImportError("Transformers library not found. Please install with 'pip install transformers'")

# Avoid circular import
from ..model_clients import ModelClient, ConfigurationError, ImageProcessingError

logger = logging.getLogger("MiniCPMVClient")

class MiniCPMVClient(ModelClient):
    """Specialized client for MiniCPMV models like AgentCPM-GUI"""
    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        """
        Initialize MiniCPMV client
        
        Args:
            model_name: Name of the model
            api_key: Optional API key (not used for offline models)
            config: Configuration dictionary
            use_component_detector: Whether to use component detection
        """
        # Log initialization
        logger.info(f"Initializing MiniCPMV client for model: {model_name}")
        
        # Set default model architecture
        if config is None:
            config = {}
            
        # Set MiniCPMV specific default configs
        if "llm_config" not in config:
            config["llm_config"] = {
                "architectures": ["MiniCPMVForCausalLM"],
                "model_type": "minicpmv",
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
        
        # Memory optimization settings
        self.max_memory = config.get("max_memory", {i: "7GB" for i in range(torch.cuda.device_count())})
        self.use_memory_optimization = config.get("use_memory_optimization", True)
        
        # Load model components
        self._load_model()
    
    def _validate_model_specific_config(self) -> None:
        """Validate MiniCPMV specific configuration"""
        if not self.config:
            raise ConfigurationError("Configuration is required for offline models")
            
        weights_path = self.config.get("weights_path")
        if not weights_path:
            raise ConfigurationError("Model weights path is required")
    
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
            
        # Try to find snapshot directory (AgentCPM-GUI)
        if "AgentCPM-GUI" in self.weights_path:
            possible_snapshot_bases = [
                os.path.join("models", "AgentCPM-GUI", "models--openbmb--AgentCPM-GUI", "snapshots"),
                os.path.join("models", self.weights_path, "models--openbmb--AgentCPM-GUI", "snapshots")
            ]
            
            for snapshot_base in possible_snapshot_bases:
                if os.path.exists(snapshot_base):
                    # Find the first subdirectory in the snapshot directory
                    for snapshot_dir in os.listdir(snapshot_base):
                        snapshot_path = os.path.join(snapshot_base, snapshot_dir)
                        if os.path.isdir(snapshot_path) and os.path.exists(os.path.join(snapshot_path, "config.json")):
                            logger.info(f"Found model snapshot path: {snapshot_path}")
                            return snapshot_path
            
        raise ConfigurationError(f"Model files not found: {self.weights_path}")
    
    def _load_model(self) -> None:
        """Load the MiniCPMV model and processor"""
        try:
            if not hasattr(self, 'model_path') or not self.model_path:
                self.model_path = self._find_model_path()
            
            logger.info(f"Loading MiniCPMV model from: {self.model_path}")
            
            # Validate model path
            if not os.path.exists(self.model_path):
                raise ConfigurationError(f"Model path does not exist: {self.model_path}")
                
            # Check config.json file
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                raise ConfigurationError(f"Config file not found at {config_path}")
                
            # Read config file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
                logger.info(f"Model type: {config_data.get('model_type', 'unknown')}")
                logger.info(f"Architectures: {config_data.get('architectures', ['unknown'])}")
            
            # Memory optimization before loading
            if self.use_memory_optimization and torch.cuda.is_available():
                torch.cuda.empty_cache()
                # Set memory allocation strategy
                os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
                logger.info("Applied memory optimization settings")
            
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
            logger.info("Loading AutoTokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load processor
            logger.info("Loading AutoProcessor...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with memory optimization
            logger.info("Loading MiniCPMVForCausalLM...")
            
            # Prepare model loading arguments with safer configuration
            model_args = {
                "torch_dtype": dtype,
                "device_map": device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True,
                "attn_implementation": "eager",
            }
            
            # Add memory constraints if optimization is enabled
            if self.use_memory_optimization and torch.cuda.is_available():
                model_args["max_memory"] = self.max_memory
                logger.info(f"Using max_memory constraints: {self.max_memory}")
            else:
                # Use sequential device mapping to avoid meta tensors
                model_args["device_map"] = "sequential"
                # Add offload configuration to prevent meta tensors
                model_args["offload_folder"] = None
            
            # Try loading with different fallback strategies
            if MINICPMV_AVAILABLE:
                try:
                    self.model = MiniCPMVForCausalLM.from_pretrained(
                        self.model_path,
                        **model_args
                    )
                    logger.info("MiniCPMVForCausalLM loaded successfully")
                    self._ensure_model_device_consistency()
                except Exception as e:
                    logger.warning(f"MiniCPMVForCausalLM loading failed: {e}")
                    # Try loading with reduced precision
                    logger.info("Attempting to load with float16 precision...")
                    model_args["torch_dtype"] = torch.float16
                    try:
                        self.model = MiniCPMVForCausalLM.from_pretrained(
                            self.model_path,
                            **model_args
                        )
                        logger.info("MiniCPMVForCausalLM loaded with float16 precision")
                        self._ensure_model_device_consistency()
                    except Exception as e2:
                        logger.warning(f"Float16 loading also failed: {e2}")
                        # Final fallback - try without device mapping and force load to prevent meta tensors
                        logger.info("Attempting to load without device mapping...")
                        model_args_fallback = {
                            "torch_dtype": torch.float16,
                            "trust_remote_code": True,
                            "low_cpu_mem_usage": True,
                            "offload_folder": None,
                            "offload_state_dict": False,
                        }
                        try:
                            self.model = MiniCPMVForCausalLM.from_pretrained(
                                self.model_path,
                                **model_args_fallback
                            )
                            # Manually move to device and ensure all components are on same device
                            if torch.cuda.is_available():
                                self.model = self.model.to("cuda")
                                # Ensure all model components are on the same device
                                self._ensure_model_device_consistency()
                            logger.info("MiniCPMVForCausalLM loaded without device mapping")
                        except Exception as e3:
                            logger.error(f"All MiniCPMVForCausalLM loading methods failed: {e3}")
                            raise e3
            else:
                logger.warning("MiniCPMVForCausalLM not available, using AutoModelForCausalLM fallback")
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    **model_args
                )
            
            logger.info("MiniCPMV model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading MiniCPMV model: {str(e)}")
            raise ConfigurationError(f"Failed to load MiniCPMV model: {str(e)}")
    
    def _resize_image_official(self, image: Image.Image) -> Image.Image:
        """
        Resize image according to AgentCPM-GUI official method
        Resize the longer side to 1120 px to save compute & memory
        """
        resolution = image.size
        w, h = resolution
        max_line_res = 1120
        if max_line_res is not None:
            max_line = max_line_res
            if h > max_line:
                w = int(w * max_line / h)
                h = max_line
            if w > max_line:
                h = int(h * max_line / w)
                w = max_line
        return image.resize((w, h), resample=Image.Resampling.LANCZOS)
    
    def _preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image for MiniCPMV"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            logger.info(f"Image size: {image.size}")
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ImageProcessingError(f"Failed to preprocess image: {str(e)}")
    
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """
        Make a prediction with MiniCPMV model
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            image_base64: Optional pre-encoded image
            
        Returns:
            Dict: Prediction results with raw_response
        """
        try:
            # Verify model is loaded
            if not hasattr(self, 'model') or self.model is None:
                logger.error("Model not loaded")
                return {"error": "Model not loaded", "raw_response": ""}
            
            # Check if image exists
            if image_path and not os.path.exists(image_path):
                alt_path = os.path.join('element_detection', os.path.basename(image_path))
                if os.path.exists(alt_path):
                    image_path = alt_path
                    logger.info(f"Using alternative image path: {alt_path}")
                else:
                    logger.error(f"Image not found: {image_path}")
                    return {"error": f"Image not found: {image_path}", "raw_response": ""}
            
            # Prepare prompt
            enhanced_prompt = self.prepare_prompt(prompt, image_path)
            
            # Preprocess image with AgentCPM-GUI resize method
            image = self._preprocess_image(image_path)
            image = self._resize_image_official(image)
            
            # Use MiniCPMV chat interface
            logger.info("Generating response with MiniCPMV")
            
            # Use AgentCPM-GUI official format
            # Build the message format as per AgentCPM-GUI documentation
            messages = [{
                "role": "user",
                "content": [
                    f"<Question>{enhanced_prompt}</Question>\nCurrent screen screenshot：",
                    image
                ]
            }]
            
            # Use the official AgentCPM-GUI system prompt for UI tasks
            SYSTEM_PROMPT = '''# Role
You are an intelligent agent familiar with Android system touch screen GUI operations, which will analyze the GUI elements and layout of the current interface based on the user's question, and generate the corresponding operation.

# Task
Based on the current screen screenshot input, output the next operation.

# Rule
- Output in compact JSON format
- The output operation must follow Schema constraints

# Schema
{"type":"object","properties":{"thought":{"type":"string","description":"Thinking process"},"POINT":{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2,"description":"Click coordinates [x,y]"},"to":{"oneOf":[{"type":"string","enum":["up","down","left","right"]},{"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":2}],"description":"Slide direction or target coordinates"},"PRESS":{"type":"string","enum":["HOME","BACK","ENTER"],"description":"Key"},"TYPE":{"type":"string","description":"Input text"},"duration":{"type":"integer","description":"Duration (milliseconds)"},"STATUS":{"type":"string","enum":["start","continue","finish","satisfied","impossible","interrupt","need_feedback"],"description":"Task status"}},"required":["thought"],"additionalProperties":false}'''
            
            # Use the model's chat method with official AgentCPM-GUI format
            try:
                # First ensure the model is in eval mode and on the correct device
                self.model.eval()
                if torch.cuda.is_available():
                    device = next(self.model.parameters()).device
                    logger.info(f"Model is on device: {device}")
                
                # Clear GPU cache before inference
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Use the model's chat method with better error handling
                with torch.no_grad():
                    response = self.model.chat(
                        image=None,  # Image is passed in msgs
                        msgs=messages,
                        system_prompt=SYSTEM_PROMPT,
                        tokenizer=self.tokenizer,
                        temperature=0.1,
                        top_p=0.3,
                        n=1,
                    )
                
                # Handle response with better validation
                if isinstance(response, str):
                    response_text = response
                elif isinstance(response, list) and len(response) > 0:
                    response_text = response[0] if isinstance(response[0], str) else str(response[0])
                else:
                    response_text = str(response)
                
                # Validate response is not empty and contains meaningful content
                if response_text and len(response_text.strip()) > 5:
                    logger.info(f"Generated response: {response_text[:100]}...")
                    return {"raw_response": response_text, "error": None}
                else:
                    logger.warning(f"Empty or invalid response: {response_text}")
                    return {"error": "Empty or invalid response generated", "raw_response": response_text}
                
            except Exception as chat_error:
                logger.error(f"Chat method failed: {chat_error}")
                
                # Check if it's a tensor-related error
                if "meta tensor" in str(chat_error).lower() or "cannot copy" in str(chat_error).lower():
                    logger.error("Tensor copying error detected - attempting model reload")
                    try:
                        # Force reload model without meta tensors
                        self._force_reload_model()
                        # Retry the chat once after reload
                        response = self.model.chat(
                            image=images[0] if images else None,
                            msgs=msgs,
                            tokenizer=self.tokenizer
                        )
                        if hasattr(response, "text"):
                            response_text = response.text
                        elif isinstance(response, str):
                            response_text = response
                        else:
                            response_text = str(response)
                        
                        return {"response": response_text, "raw_response": response_text}
                    except Exception as retry_error:
                        logger.error(f"Model reload and retry failed: {retry_error}")
                        return {"error": f"Model weights error: {str(chat_error)}", "raw_response": ""}
                else:
                    return {"error": f"Generation failed: {str(chat_error)}", "raw_response": ""}
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}", "raw_response": ""}
            
    def _force_reload_model(self) -> None:
        """Force reload model without meta tensors to fix tensor copying errors"""
        logger.info("Force reloading model to fix meta tensor issues...")
        
        # Clear current model
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reload with minimal args to avoid meta tensors
        reload_args = {
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "device_map": None,  # No device mapping
            "offload_folder": None,
            "offload_state_dict": False,
        }
        
        if MINICPMV_AVAILABLE:
            self.model = MiniCPMVForCausalLM.from_pretrained(
                self.model_path,
                **reload_args
            )
            # Manually move to CUDA if available
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
        else:
            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                **reload_args
            )
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
                self._ensure_model_device_consistency()
        
        logger.info("Model reloaded successfully")
    
    def _ensure_model_device_consistency(self) -> None:
        """Ensure all model components are on the same device to prevent tensor device mismatch"""
        try:
            target_device = self.device
            
            # Move core model components to target device
            if hasattr(self.model, 'llm') and self.model.llm is not None:
                self.model.llm = self.model.llm.to(target_device)
            
            if hasattr(self.model, 'vpm') and self.model.vpm is not None:
                self.model.vpm = self.model.vpm.to(target_device)
                
            if hasattr(self.model, 'resampler') and self.model.resampler is not None:
                self.model.resampler = self.model.resampler.to(target_device)
                
            # Ensure tokenizer embeddings are also on target device
            if hasattr(self.model, 'embed_tokens') and self.model.embed_tokens is not None:
                self.model.embed_tokens = self.model.embed_tokens.to(target_device)
                
            logger.info(f"All model components moved to {target_device}")
            
        except Exception as e:
            logger.warning(f"Failed to ensure device consistency: {e}")

    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        super().cleanup()