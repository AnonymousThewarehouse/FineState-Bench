#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import hashlib
import logging
import torch
import sys
from typing import Dict, Optional
from PIL import Image

try:
    from transformers import (
        AutoTokenizer, AutoProcessor, AutoModelForCausalLM, 
        AutoModel, AutoModelForVision2Seq
    )
    # Try to import Qwen2VL related classes
    try:
        from transformers import Qwen2VLForConditionalGeneration
        QWEN2VL_AVAILABLE = True
    except ImportError:
        QWEN2VL_AVAILABLE = False
except ImportError:
    print("Warning: transformers library import failed, please ensure it is installed")

from ..model_clients import ModelClient, ConfigurationError, ImageProcessingError

logger = logging.getLogger("OfflineModelClient")

class OfflineModelClient(ModelClient):
    """Offline model client implementation"""
    
    _image_cache = {}  # Class-level image cache
    _prompt_cache = {}  # Class-level prompt cache
    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        """
        Initialize offline model client
        
        Args:
            model_name: Name of the model
            api_key: Not used for offline models
            config: Configuration dictionary
            use_component_detector: Whether to use component detection
        """
        # Call parent constructor, which will initialize the logger
        super().__init__(model_name, api_key, config, use_component_detector)
        
        # Initialize model attributes
        self.model = None
        self.tokenizer = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.weights_path = self.config.get("weights_path", "")
        self.model_params = self.config.get("params", {})
        self.image_processing = self.model_params.get("image_processing", {})
        
        # Load model components
        self._load_model()
        
    def _validate_model_specific_config(self) -> None:
        """Validate offline model specific configuration"""
        if not self.config:
            raise ConfigurationError("Configuration is required for offline models")
            
        weights_path = self.config.get("weights_path")
        if not weights_path:
            raise ConfigurationError("Model weights path is required")
            
        # Check and set default vision_config if needed
        vision_config = self.config.get("vision_config")
        if not vision_config:
            logger.warning(f"Vision model configuration missing for {self.model_name}, using default values")
            self.config["vision_config"] = {
                "num_channels": 3, "patch_size": 14, "image_size": 224, "qkv_bias": False,
                "hidden_size": 1536, "num_attention_heads": 24, "intermediate_size": 6144,
                "qk_normalization": True, "num_hidden_layers": 24, "use_flash_attn": True,
                "hidden_act": "gelu", "norm_type": "rms_norm", "layer_norm_eps": 1e-6
            }
            
        # Check and set default llm_config if needed
        llm_config = self.config.get("llm_config")
        architecture = "Qwen2_5VLForCausalLM"
        if "os-atlas" in self.model_name.lower():
            architecture = "Phi3ForCausalLM"
            
        if not llm_config:
            logger.warning(f"Language model configuration missing for {self.model_name}, using default values")
            self.config["llm_config"] = {
                "architectures": [architecture], "vocab_size": 151936, "hidden_size": 2560,
                "intermediate_size": 6912, "num_hidden_layers": 32, "num_attention_heads": 32,
                "max_position_embeddings": 4096, "rms_norm_eps": 1e-6, "use_cache": True,
                "rope_theta": 10000.0
            }
        elif "architectures" not in llm_config:
            logger.warning(f"Model architecture not specified for {self.model_name}, using default")
            self.config["llm_config"]["architectures"] = [architecture]
            
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
            
        # Try to find snapshot directory pattern for HuggingFace models
        possible_paths = [
            os.path.join("models", os.path.basename(self.weights_path)),  # e.g., models/UGround-V1-7B
            os.path.join("models", self.weights_path)  # e.g., models/osunlp/UGround-V1-7B
        ]
        
        for base_path in possible_paths:
            if os.path.exists(base_path):
                # Look for HuggingFace snapshot pattern
                for item in os.listdir(base_path):
                    item_path = os.path.join(base_path, item)
                    if os.path.isdir(item_path) and "models--" in item:
                        # Found HuggingFace models directory
                        snapshots_path = os.path.join(item_path, "snapshots")
                        if os.path.exists(snapshots_path):
                            # Find the first snapshot
                            for snapshot in os.listdir(snapshots_path):
                                snapshot_path = os.path.join(snapshots_path, snapshot)
                                if os.path.isdir(snapshot_path) and os.path.exists(os.path.join(snapshot_path, "config.json")):
                                    logger.info(f"Found model snapshot path: {snapshot_path}")
                                    return snapshot_path
            
        raise ConfigurationError(f"Model files not found: {self.weights_path}")
        
    def _load_model(self) -> None:
        """Load the model and tokenizer"""
        try:
            if not hasattr(self, 'model_path') or not self.model_path:
                self.model_path = self._find_model_path()
                
            self.logger.info(f"Starting to load model: {self.model_path}")
            
            # Load tokenizer
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            
            # Load model
            self.logger.info("Loading model...")
            model_name_norm = os.path.basename(os.path.normpath(self.model_path))
            
            # Special handling for UGround models
            if "uground" in model_name_norm.lower():
                model_loading_strategies = []
                
                # If Qwen2VLForConditionalGeneration is available, use it first
                if QWEN2VL_AVAILABLE:
                    model_loading_strategies.extend([
                        # Use weights_only mode to bypass security restrictions
                        lambda: Qwen2VLForConditionalGeneration.from_pretrained(
                            self.model_path, 
                            trust_remote_code=True, 
                            torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                            device_map="auto",
                            attn_implementation="eager",
                            weights_only=False  # Bypass security restrictions
                        ),
                        # Simplified version, no special attention settings
                        lambda: Qwen2VLForConditionalGeneration.from_pretrained(
                            self.model_path, 
                            trust_remote_code=True, 
                            torch_dtype=torch.float16,
                            device_map="auto",
                            weights_only=False
                        ),
                        # CPU mode alternative
                        lambda: Qwen2VLForConditionalGeneration.from_pretrained(
                            self.model_path, 
                            trust_remote_code=True, 
                            torch_dtype=torch.float32,
                            device_map="cpu",
                            weights_only=False
                        ),
                    ])
                
                # Alternative strategies
                model_loading_strategies.extend([
                    lambda: AutoModelForCausalLM.from_pretrained(
                        self.model_path, 
                        trust_remote_code=True, 
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
                        device_map="auto",
                        attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
                    ),
                    lambda: AutoModelForCausalLM.from_pretrained(
                        self.model_path, 
                        trust_remote_code=True, 
                        torch_dtype=torch.float16, 
                        device_map="auto"
                    ),
                    lambda: AutoModelForVision2Seq.from_pretrained(
                        self.model_path, 
                        trust_remote_code=True, 
                        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, 
                        device_map="auto"
                    ),
                ])
            else:
                # General model loading strategies
                model_loading_strategies = [
                    lambda: AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"),
                    lambda: AutoModel.from_pretrained(self.model_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="auto"),
                    lambda: AutoModelForVision2Seq.from_pretrained(self.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16, device_map="auto"),
                    lambda: AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True, low_cpu_mem_usage=True, device_map="auto")
                ]
            
            for i, strategy in enumerate(model_loading_strategies):
                try:
                    self.model = strategy()
                    self.logger.info(f"Model loaded successfully with strategy #{i+1}")
                    
                    # Verify if the model has a generate method
                    if not hasattr(self.model, 'generate'):
                        self.logger.warning(f"Model loaded with strategy #{i+1} does not have 'generate' method")
                        if i < len(model_loading_strategies) - 1:
                            continue  # Try next strategy
                        else:
                            raise ConfigurationError(f"No model loading strategy provided a model with 'generate' method for {model_name_norm}")
                    
                    break
                except Exception as e:
                    self.logger.warning(f"Model loading strategy #{i+1} failed: {e}")
                    if i == len(model_loading_strategies) - 1:
                        raise ConfigurationError(f"All model loading strategies failed for {model_name_norm}")

            # Load processor if applicable
            try:
                self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
                self.logger.info("Processor loaded successfully.")
            except Exception as e:
                self.logger.warning(f"Could not load AutoProcessor: {e}. This might be normal for some models.")
                
        except Exception as e:
            self.logger.error(f"Error during model loading process: {e}")
            raise ConfigurationError(f"Failed to load model '{self.model_name}': {e}")
            
    def _verify_model_loaded(self) -> bool:
        """Verify that the model and tokenizer are properly loaded"""
        if not all([hasattr(self, 'model'), self.model, hasattr(self, 'tokenizer'), self.tokenizer]):
            self.logger.error("Model or tokenizer not loaded.")
            return False
        return True
            
    def _load_and_cache_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and cache image with processing"""
        try:
            cache_key = hashlib.md5(image_path.encode()).hexdigest()
            if cache_key in self._image_cache:
                self.logger.debug(f"Using cached image: {image_path}")
                return self._image_cache[cache_key]
                
            if not os.path.exists(image_path):
                alt_path = os.path.join('element_detection', os.path.basename(image_path))
                image_path = alt_path if os.path.exists(alt_path) else image_path
            
            if not os.path.exists(image_path):
                 raise ImageProcessingError(f'Image not found: {image_path}')

            image = Image.open(image_path).convert('RGB')
            
            # Simplified image resizing logic
            if self.image_processing.get("do_resize", True):
                max_pixels = self.image_processing.get("max_pixels", 1048576)
                if image.size[0] * image.size[1] > max_pixels:
                    ratio = (max_pixels / (image.size[0] * image.size[1])) ** 0.5
                    new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
                    image = image.resize(new_size, Image.LANCZOS)
            
            if len(self._image_cache) < 50:
                self._image_cache[cache_key] = image
                
            self.logger.info(f"Loaded image: {image_path}, size: {image.size}")
            return image
        except Exception as e:
            raise ImageProcessingError(f"Failed to load and process image: {str(e)}")
            
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """Make a prediction with the offline model"""
        try:
            if not self._verify_model_loaded():
                return {"error": "Model components not properly loaded", "raw_response": ""}
                
            image = self._load_and_cache_image(image_path) if image_path else None
            enhanced_prompt = self.prepare_prompt(prompt, image_path)
            
            model_name = self.model_name.lower()
            
            # Qwen-VL specific chat method
            if "qwen" in model_name and "vl" in model_name and hasattr(self.model, 'chat'):
                try:
                    self.logger.info("Using Qwen-VL chat method")
                    messages = [{"role": "user", "content": [{"image": image_path}, {"text": enhanced_prompt}]}]
                    response, _ = self.model.chat(self.tokenizer, messages=messages, history=None)
                    return {"raw_response": response}
                except Exception as e:
                    self.logger.error(f"Qwen-VL chat method failed: {e}")
            
            # Standard generation
            return self._standard_generate(enhanced_prompt, image)
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return {"error": f"Prediction failed: {e}", "raw_response": ""}
            
    def _standard_generate(self, prompt: str, image: Optional[Image.Image]) -> Dict:
        """Generate response using standard methods."""
        self.logger.info("Using standard generation method")
        try:
            # Special handling for UGround models
            if "uground" in self.model_name.lower():
                return self._uground_generate(prompt, image)
            
            if self.processor and image:
                inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
            else:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            outputs = self.model.generate(**inputs, max_new_tokens=512)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            return {"raw_response": response}
        except Exception as e:
            self.logger.error(f"Standard generation failed: {e}")
            return {"error": f"Standard generation failed: {e}", "raw_response": ""}
    
    def _uground_generate(self, prompt: str, image: Optional[Image.Image]) -> Dict:
        """UGround specific generation method"""
        try:
            self.logger.info("Using UGround-specific generation method")
            
            # Ensure processor and image are available
            if not self.processor or not image:
                return {"error": "UGround requires both processor and image", "raw_response": ""}
            
            # UGround specific input processing - use Qwen2VL messages format
            try:
                # UGround is based on Qwen2VL, use messages format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image
                            },
                            {
                                "type": "text", 
                                "text": prompt
                            }
                        ]
                    }
                ]
                
                # Apply chat template
                if hasattr(self.processor, 'apply_chat_template') and hasattr(self.processor, 'chat_template') and self.processor.chat_template:
                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    self.logger.info("Using UGround processor chat template")
                else:
                    # If no chat template, raise exception to enter fallback
                    raise ValueError("No chat template available")
                
                # Process image and text
                try:
                    # Import qwen_vl_utils for processing
                    from qwen_vl_utils import process_vision_info
                    image_inputs, video_inputs = process_vision_info(messages)
                except ImportError:
                    # Alternative: use image directly
                    image_inputs = [image]
                    video_inputs = None
                
                # Use processor to process input
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
                
                self.logger.info(f"UGround chat template - input_ids shape: {inputs.input_ids.shape}")
                if hasattr(inputs, 'pixel_values') and inputs.pixel_values is not None:
                    self.logger.info(f"UGround chat template - pixel_values shape: {inputs.pixel_values.shape}")
                
            except Exception as e:
                self.logger.warning(f"UGround chat template failed, trying alternative: {e}")
                # Fallback to manually constructing chat format
                try:
                    # Manually construct Qwen2VL format text
                    formatted_text = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>\n<|im_start|>assistant\n"
                    
                    # Use processor to process image and text
                    inputs = self.processor(
                        text=[formatted_text],
                        images=[image],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    self.logger.info(f"UGround manual format - input_ids shape: {inputs.input_ids.shape}")
                    
                except Exception as e2:
                    self.logger.warning(f"Manual format failed, trying basic processing: {e2}")
                    # Final fallback: basic text processing, but add image token
                    try:
                        # Directly process image and text, no special format
                        inputs = self.processor(
                            text=prompt,
                            images=image,
                            return_tensors="pt"
                        ).to(self.device)
                        
                        self.logger.info(f"UGround basic processing - input_ids shape: {inputs.input_ids.shape}")
                        
                    except Exception as e3:
                        self.logger.error(f"All UGround input processing methods failed: {e3}")
                        return {"error": f"UGround input processing failed: {e3}", "raw_response": ""}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
                )
            
            # Decode response
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
            ]
            
            response = self.tokenizer.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            self.logger.info(f"UGround generated response length: {len(response)}")
            return {"raw_response": response}
            
        except Exception as e:
            self.logger.error(f"UGround generation failed: {e}")
            return {"error": f"UGround generation failed: {e}", "raw_response": ""}

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all caches and free memory"""
        cls._image_cache.clear()
        cls._prompt_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.getLogger(cls.__name__).info("Cleared model caches")
        
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        super().cleanup() 