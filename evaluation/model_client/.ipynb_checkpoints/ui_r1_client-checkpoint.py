#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import re
import torch
import math
from typing import Dict, Optional, List, Union, Any, Tuple
from PIL import Image
import sys

# Make sure required libraries are available
try:
    from transformers import AutoProcessor, AutoTokenizer
    # UI-R1 models use Qwen2_5_VL (note the underscore)
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        QWEN2_5_VL_AVAILABLE = True
    except ImportError:
        QWEN2_5_VL_AVAILABLE = False
        # We'll log this warning later when the logger is initialized
except ImportError:
    raise ImportError("Transformers library not found. Please install with 'pip install transformers'")

# UI-R1 specific dependencies
try:
    from qwen_vl_utils import process_vision_info
    QWEN_VL_UTILS_AVAILABLE = True
except ImportError:
    QWEN_VL_UTILS_AVAILABLE = False
    print("Warning: qwen_vl_utils not available. Please install with 'pip install qwen-vl-utils'")

# 避免循环导入
from ..model_clients import ModelClient, ConfigurationError, ImageProcessingError

logger = logging.getLogger("UIR1Client")

# UI-R1 specific constants from HuggingFace example
MAX_IMAGE_PIXELS = 12845056  # UI-R1 specific max pixels

def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """
    Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    
    From Qwen2VL implementation for UI-R1 models
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar

class UIR1Client(ModelClient):
    """Universal client for UI-R1 series models (Qwen2.5-VL-3B-UI-R1, GUI-R1, Jedi, etc.)"""
    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        """
        Initialize UI-R1 client
        
        Args:
            model_name: Name of the model
            api_key: Optional API key (not used for offline models)
            config: Configuration dictionary
            use_component_detector: Whether to use component detection
        """
        # Log initialization
        logger.info(f"Initializing UI-R1 client for model: {model_name}")
        
        # Set default model architecture
        if config is None:
            config = {}
            
        # Detect model variant
        self.model_variant = self._detect_model_variant(model_name)
        logger.info(f"Detected model variant: {self.model_variant}")
            
        # Set UI-R1 specific default configs
        if "llm_config" not in config:
            config["llm_config"] = {
                "architectures": ["Qwen2_5_VLForConditionalGeneration"],
                "model_type": "qwen2_5_vl",
                "trust_remote_code": True,
                "attn_implementation": "flash_attention_2"  # UI-R1 specific
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
    
    def _detect_model_variant(self, model_name: str) -> str:
        """Detect UI-R1 model variant from model name"""
        model_name_lower = model_name.lower()
        
        # Check for GUI-R1 first (more specific)
        if "gui-r1" in model_name_lower:
            return "gui-r1"
        # Check for UI-R1-E variant
        elif "ui-r1-e" in model_name_lower or "ui-r1-3b-e" in model_name_lower:
            return "ui-r1-e"
        # Check for general UI-R1
        elif "ui-r1" in model_name_lower:
            return "ui-r1"
        # Check for Jedi models
        elif "jedi" in model_name_lower:
            return "jedi"
        else:
            # Default to ui-r1 for Qwen2.5-VL-UI models
            return "ui-r1"
    
    def _validate_model_specific_config(self) -> None:
        """Validate UI-R1 specific configuration"""
        if not self.config:
            raise ConfigurationError("Configuration is required for offline models")
            
        weights_path = self.config.get("weights_path")
        if not weights_path:
            raise ConfigurationError("Model weights path is required")
            
        # Additional validation for UI-R1 models
        if not QWEN_VL_UTILS_AVAILABLE:
            logger.warning("qwen_vl_utils not available, will use fallback processing")
    
    def _find_model_path(self) -> str:
        """Find the model path in various locations for different UI-R1 variants"""
        # 首先尝试直接路径
        if os.path.exists(self.weights_path):
            logger.info(f"Using local model path: {self.weights_path}")
            return self.weights_path
            
        # 尝试models目录
        local_path = os.path.join("models", self.weights_path)
        if os.path.exists(local_path):
            logger.info(f"Found cached model path: {local_path}")
            return local_path
            
        # 尝试查找快照目录 - 支持不同模型变体
        variant_paths = []
        
        if "Qwen2.5-VL" in self.weights_path:
            # Qwen2.5-VL-3B-UI-R1 系列
            variant_paths.extend([
                os.path.join("models", self.weights_path, "models--LZXzju--Qwen2.5-VL-3B-UI-R1", "snapshots"),
                os.path.join("models", "Qwen2.5-VL-3B-UI-R1", "models--LZXzju--Qwen2.5-VL-3B-UI-R1", "snapshots"),
                os.path.join("models", self.weights_path, "models--LZXzju--Qwen2.5-VL-3B-UI-R1-E", "snapshots"),
                os.path.join("models", "Qwen2.5-VL-3B-UI-R1-E", "models--LZXzju--Qwen2.5-VL-3B-UI-R1-E", "snapshots"),
            ])
        elif "GUI-R1" in self.weights_path:
            # GUI-R1 系列
            variant_paths.extend([
                os.path.join("models", self.weights_path, "models--ritzzai--GUI-R1", "snapshots"),
                os.path.join("models", "GUI-R1-3B", "models--ritzzai--GUI-R1", "snapshots"),
                os.path.join("models", "GUI-R1-7B", "models--ritzzai--GUI-R1", "snapshots"),
            ])
        elif "Jedi" in self.weights_path:
            # Jedi 系列
            variant_paths.extend([
                os.path.join("models", self.weights_path, "models--xlangai--Jedi-3B-1080p", "snapshots"),
                os.path.join("models", "Jedi-3B-1080p", "models--xlangai--Jedi-3B-1080p", "snapshots"),
                os.path.join("models", self.weights_path, "models--xlangai--Jedi-7B-1080p", "snapshots"),
                os.path.join("models", "Jedi-7B-1080p", "models--xlangai--Jedi-7B-1080p", "snapshots"),
            ])
        
        # 通用路径
        variant_paths.append(os.path.join("models", self.weights_path, "snapshots"))
        
        for snapshot_base in variant_paths:
            if os.path.exists(snapshot_base):
                # 查找快照目录中的第一个子目录
                for snapshot_dir in os.listdir(snapshot_base):
                    snapshot_path = os.path.join(snapshot_base, snapshot_dir)
                    if os.path.isdir(snapshot_path):
                        # 检查是否直接有config.json文件
                        if os.path.exists(os.path.join(snapshot_path, "config.json")):
                            logger.info(f"Found model snapshot path: {snapshot_path}")
                            return snapshot_path
                        # 检查是否有子目录包含配置文件（如GUI-R1的情况）
                        for sub_dir in os.listdir(snapshot_path):
                            sub_path = os.path.join(snapshot_path, sub_dir)
                            if os.path.isdir(sub_path) and os.path.exists(os.path.join(sub_path, "config.json")):
                                logger.info(f"Found model snapshot path in subdirectory: {sub_path}")
                                return sub_path
            
        raise ConfigurationError(f"Model files not found: {self.weights_path}")
    
    def _load_model(self) -> None:
        """Load the UI-R1 model and processor"""
        try:
            if not hasattr(self, 'model_path') or not self.model_path:
                self.model_path = self._find_model_path()
            
            logger.info(f"Loading UI-R1 model from: {self.model_path}")
            
            # Validate model path
            if not os.path.exists(self.model_path):
                raise ConfigurationError(f"Model path does not exist: {self.model_path}")
                
            # 检查config.json文件
            config_path = os.path.join(self.model_path, "config.json")
            if not os.path.exists(config_path):
                raise ConfigurationError(f"Config file not found at {config_path}")
                
            # 读取配置文件
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
            
            # Load processor with UI-R1 specific settings
            logger.info("Loading AutoProcessor with UI-R1 settings...")
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load model with UI-R1 specific settings
            logger.info("Loading Qwen2_5_VLForConditionalGeneration...")
            if QWEN2_5_VL_AVAILABLE:
                # Try with flash_attention_2 first, fallback if not available
                try:
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        attn_implementation="flash_attention_2",  # UI-R1 specific
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
            
            # Create a new minimal generation config to replace the problematic one
            if hasattr(self.model, 'generation_config'):
                from transformers import GenerationConfig
                # Create a completely new minimal generation config
                self.model.generation_config = GenerationConfig(
                    bos_token_id=self.tokenizer.bos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=False,
                    use_cache=False,
                    max_new_tokens=64
                )
                logger.info("Replaced generation config with minimal settings")
            
            logger.info("UI-R1 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading UI-R1 model: {str(e)}")
            raise ConfigurationError(f"Failed to load UI-R1 model: {str(e)}")
    
    def _preprocess_image_ui_r1(self, image_path: str) -> Tuple[Image.Image, Tuple[float, float]]:
        """
        Preprocess image with UI-R1 specific logic including smart_resize
        
        Returns:
            Tuple of (processed_image, scale_factors)
        """
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            origin_width, origin_height = image.size
            
            # Use smart_resize from UI-R1 implementation
            resized_height, resized_width = smart_resize(
                origin_height, origin_width, max_pixels=self.max_image_pixels
            )
            
            logger.info(f"Original size: {origin_width}x{origin_height}, UI-R1 resized to: {resized_width}x{resized_height}")
            
            # Calculate scale factors for coordinate rescaling
            scale_x = origin_width / resized_width
            scale_y = origin_height / resized_height
            
            # Resize image if needed
            if resized_width != origin_width or resized_height != origin_height:
                image = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
                
            return image, (scale_x, scale_y)
            
        except Exception as e:
            logger.error(f"Error preprocessing image with UI-R1 logic: {str(e)}")
            raise ImageProcessingError(f"Failed to preprocess image: {str(e)}")
    
    def _generate_ui_r1_prompt(self, task_prompt: str) -> str:
        """Generate simpler prompt to avoid timeout issues"""
        # Simplified prompt format to avoid generation issues
        simplified_prompt = f"Task: {task_prompt}\nPlease provide coordinates in [x, y] format."
        return simplified_prompt
    
    def _extract_coordinates_from_response(self, response: str, scale_factors: Tuple[float, float]) -> Optional[List[int]]:
        """
        Extract coordinates from UI-R1 response format and apply scaling
        
        Args:
            response: Model response containing <answer> tags
            scale_factors: (scale_x, scale_y) for coordinate rescaling
            
        Returns:
            List of scaled coordinates [x, y] or None if not found
        """
        try:
            # Extract content from <answer> tags
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if not answer_match:
                logger.warning("No <answer> tags found in response")
                return None
            
            answer_content = answer_match.group(1).strip()
            
            # Extract coordinates using regex
            coord_match = re.search(r'\[(\d+),\s*(\d+)\]', answer_content)
            if coord_match:
                x = int(coord_match.group(1))
                y = int(coord_match.group(2))
                
                # Apply scaling factors
                scale_x, scale_y = scale_factors
                scaled_x = int(x * scale_x)
                scaled_y = int(y * scale_y)
                
                logger.info(f"Extracted coordinates: ({x}, {y}) -> scaled: ({scaled_x}, {scaled_y})")
                return [scaled_x, scaled_y]
            
            logger.warning(f"No coordinates found in answer: {answer_content}")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting coordinates: {str(e)}")
            return None
    
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """
        Make a prediction with UI-R1 model
        
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
            
            # Use simple prompt like InfiGUI (which works)
            enhanced_prompt = self.prepare_prompt(prompt, image_path)
            
            # Preprocess image with UI-R1 logic
            processed_image, scale_factors = self._preprocess_image_ui_r1(image_path)
            
            # Use InfiGUI-style message format since it works
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": enhanced_prompt},
                    ],
                }
            ]
            
            # Process the input using UI-R1 approach
            logger.info("Preparing inference inputs for UI-R1")
            try:
                # Apply chat template
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Chat template not available, using manual format: {e}")
                # Manual format for UI-R1 (similar to Qwen2.5-VL)
                text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{ui_r1_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            # Process vision info using qwen_vl_utils
            try:
                if QWEN_VL_UTILS_AVAILABLE:
                    image_inputs, video_inputs = process_vision_info(messages)
                    logger.info(f"Processed vision info: {len(image_inputs) if image_inputs else 0} images")
                else:
                    # Fallback: load image directly
                    image_inputs = [processed_image]
                    video_inputs = None
                    logger.info("Using fallback image loading")
            except Exception as e:
                logger.error(f"Error processing vision info: {e}")
                # Fallback: load image directly
                image_inputs = [processed_image]
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
            logger.info("Starting generation with overridden config")
            
            with torch.no_grad():
                try:
                    # Use the model's generation config which we've replaced
                    logger.info("Using model's generation config for generation")
                    
                    # Add timeout mechanism using multiprocessing
                    import signal
                    import functools
                    
                    def timeout_handler(signum, frame):
                        raise TimeoutError("Generation timeout after 30 seconds")
                    
                    # Set timeout signal
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(30)  # 30 seconds timeout
                    
                    try:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=64,  # 增加输出长度以包含完整坐标
                            do_sample=False,
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            use_cache=True,
                        )
                        
                        # Cancel the alarm
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                        
                        logger.info("Generation completed successfully")
                        
                    except TimeoutError as timeout_error:
                        signal.alarm(0)
                        signal.signal(signal.SIGALRM, old_handler)
                        logger.error(f"Generation timeout: {timeout_error}")
                        return {"error": f"Generation timeout: {str(timeout_error)}", "raw_response": ""}
                    
                except Exception as gen_error:
                    logger.error(f"Generation failed: {gen_error}")
                    return {"error": f"Generation failed: {str(gen_error)}", "raw_response": ""}
            
            # Process output
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            logger.info(f"Generated response: {output_text[:100]}...")
            
            # Extract coordinates if present
            coordinates = self._extract_coordinates_from_response(output_text, scale_factors)
            
            # Add coordinates to response if extracted
            result = {"raw_response": output_text, "error": None}
            if coordinates:
                result["coordinates"] = coordinates
                
            return result
            
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