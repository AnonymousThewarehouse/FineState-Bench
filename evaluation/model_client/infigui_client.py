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
    # InfiGUI uses Qwen2_5_VL (note the underscore)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        QWEN2_5_VL_AVAILABLE = True
    except ImportError:
        QWEN2_5_VL_AVAILABLE = False
        # We'll log this warning later when the logger is initialized
except ImportError:
    raise ImportError("Transformers library not found. Please install with 'pip install transformers'")

# InfiGUI specific dependencies
try:
    from qwen_vl_utils import process_vision_info, smart_resize
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("Warning: qwen_vl_utils not available. Please install with 'pip install qwen-vl-utils'")

# Avoid circular import
from ..model_clients import ModelClient, ConfigurationError, ImageProcessingError

logger = logging.getLogger("InfiGUIClient")

# InfiGUI specific constants from HuggingFace example
MAX_IMAGE_PIXELS = 5600 * 28 * 28

class InfiGUIClient(ModelClient):
    """Specialized client for InfiGUI-R1-3B models"""
    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        """
        Initialize InfiGUI client
        
        Args:
            model_name: Name of the model
            api_key: Optional API key (not used for offline models)
            config: Configuration dictionary
            use_component_detector: Whether to use component detection
        """
        # Log initialization
        logger.info(f"Initializing InfiGUI client for model: {model_name}")
        
        # Set default model architecture
        if config is None:
            config = {}
            
        # Set InfiGUI specific default configs
        if "llm_config" not in config:
            config["llm_config"] = {
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "model_type": "qwen2_5_vl",
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2"  # InfiGUI specific
            }
        
        # Call parent constructor
        super().__init__(model_name, api_key, config, use_component_detector)
        
        # Initialize model attributes
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = config.get("weights_path", "")
        self.max_image_pixels = config.get("max_image_pixels", MAX_IMAGE_PIXELS)
        
        # Load model components
        self._load_model()
    
    def _validate_model_specific_config(self) -> None:
        """Validate InfiGUI specific configuration"""
        if not self.config:
            raise ConfigurationError("Configuration is required for offline models")
            
        weights_path = self.config.get("weights_path")
        if not weights_path:
            raise ConfigurationError("Model weights path is required")
            
        # Additional validation for InfiGUI models
        if not QWEN_VL_UTILS_AVAILABLE:
            raise ConfigurationError("qwen_vl_utils is required for InfiGUI models, please install with 'pip install qwen-vl-utils'")
    
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
            
        # Try to find snapshot directory (InfiGUI models)
        if "InfiGUI" in self.weights_path:
            # Try different snapshot directory structures
            possible_snapshot_bases = [
                os.path.join("models", self.weights_path, "models--Reallm-Labs--InfiGUI-R1-3B", "snapshots"),
                os.path.join("models", "InfiGUI-R1-3B", "models--Reallm-Labs--InfiGUI-R1-3B", "snapshots")
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
        """Load the InfiGUI model and processor"""
        try:
            if not hasattr(self, 'model_path') or not self.model_path:
                self.model_path = self._find_model_path()
            
            logger.info(f"Loading InfiGUI model from: {self.model_path}")
            
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
            
            # Load processor with InfiGUI specific settings
            logger.info("Loading AutoProcessor with InfiGUI settings...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                max_pixels=self.max_image_pixels,
                padding_side="left"  # InfiGUI specific
            )
            
            # Load model with InfiGUI specific settings
            logger.info("Loading Qwen2_5_VLForConditionalGeneration...")
            if QWEN2_5_VL_AVAILABLE:
                # Try with flash_attention_2 first, fallback if not available
                try:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        attn_implementation="flash_attention_2",  # InfiGUI specific
                        device_map=device_map,
                        trust_remote_code=True
                    )
                    logger.info("Successfully loaded with flash_attention_2")
                except ImportError as e:
                    if "flash_attn" in str(e):
                        logger.warning("flash_attn not available, falling back to eager attention")
                        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            self.model_path,
                            torch_dtype=dtype,
                            attn_implementation="eager",  # Fallback
                            device_map=device_map,
                            trust_remote_code=True
                        )
                        logger.info("Successfully loaded with eager attention")
                    else:
                        raise e
            else:
                logger.warning("Qwen2_5_VL not available, using AutoModelForCausalLM fallback")
                from transformers import AutoModelForCausalLM
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    torch_dtype=dtype,
                    device_map=device_map,
                    trust_remote_code=True
                )
            
            logger.info("InfiGUI model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading InfiGUI model: {str(e)}")
            raise ConfigurationError(f"Failed to load InfiGUI model: {str(e)}")
    
    def _preprocess_image(self, image_path: str) -> Image.Image:
        """Preprocess image with InfiGUI specific logic"""
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            width, height = image.size
            
            # Use smart_resize from qwen_vl_utils
            if QWEN_VL_UTILS_AVAILABLE:
                new_height, new_width = smart_resize(height, width, max_pixels=self.max_image_pixels)
                logger.info(f"Original size: {width}x{height}, Resized to: {new_width}x{new_height}")
                
                # Resize image if needed
                if new_width != width or new_height != height:
                    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            else:
                logger.warning("smart_resize not available, using original image size")
                
            return image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            raise ImageProcessingError(f"Failed to preprocess image: {str(e)}")
    
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """
        Make a prediction with InfiGUI model
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            image_base64: Optional pre-encoded image
            
        Returns:
            Dict: Prediction results with raw_response
        """
        try:
            # Verify model is loaded
            if not hasattr(self, 'model') or self.model is None or not hasattr(self, 'processor') or self.processor is None:
                logger.error("Model or processor not loaded")
                return {"error": "Model components not loaded", "raw_response": ""}
            
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
            
            # Preprocess image
            image = self._preprocess_image(image_path)
            
            # Format input in the InfiGUI expected format (similar to Qwen2.5-VL)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": enhanced_prompt},
                    ],
                }
            ]
            
            # Process the input using InfiGUI approach
            logger.info("Preparing inference inputs for InfiGUI")
            try:
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template not available, using manual format: {e}")
                # Manual format for InfiGUI (similar to Qwen2.5-VL)
                text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{enhanced_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Process vision info using qwen_vl_utils
            try:
                if QWEN_VL_UTILS_AVAILABLE:
                    image_inputs, video_inputs = process_vision_info(messages)
                    logger.info(f"Processed vision info: {len(image_inputs) if image_inputs else 0} images")
                else:
                    # Fallback: load image directly
                    image_inputs = [image]
                    video_inputs = None
                    logger.info("Using fallback image loading")
            except Exception as e:
                logger.error(f"Error processing vision info: {e}")
                # Fallback: load image directly
                image_inputs = [image]
                video_inputs = None
                logger.info("Using fallback image loading")
            
            # Create model inputs
            try:
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                )
                logger.info("Successfully created model inputs")
            except Exception as e:
                logger.error(f"Error creating model inputs: {e}")
                return {"error": f"Failed to create model inputs: {e}", "raw_response": ""}
            
            # Move inputs to the correct device
            device = next(self.model.parameters()).device
            inputs = inputs.to(device)
            
            # Generate response
            logger.info("Generating response")
            # Prepare generation parameters (avoid unsupported parameters)
            generation_kwargs = {
                "max_new_tokens": self.config.get("params", {}).get("max_tokens", 512),
                "do_sample": self.config.get("params", {}).get("do_sample", False),
            }
            
            # Only add sampling parameters if do_sample is True
            if generation_kwargs["do_sample"]:
                generation_kwargs["temperature"] = self.config.get("params", {}).get("temperature", 0.1)
                generation_kwargs["top_p"] = self.config.get("params", {}).get("top_p", 0.9)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Process output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Generated response: {output_text[:100]}...")
            return {"raw_response": output_text, "error": None}
            
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