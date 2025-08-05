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
    from transformers import AutoProcessor, AutoTokenizer, pipeline
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

# Avoid circular import
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
        # Try direct path first
        if os.path.exists(self.weights_path):
            logger.info(f"Using local model path: {self.weights_path}")
            return self.weights_path
            
        # Try models directory
        local_path = os.path.join("models", self.weights_path)
        if os.path.exists(local_path):
            logger.info(f"Found cached model path: {local_path}")
            return local_path
            
        # Try to find snapshot directory - support different model variants
        variant_paths = []
        
        if "Qwen2.5-VL" in self.weights_path:
            # Qwen2.5-VL-3B-UI-R1 series
            variant_paths.extend([
                os.path.join("models", self.weights_path, "models--LZXzju--Qwen2.5-VL-3B-UI-R1", "snapshots"),
                os.path.join("models", "Qwen2.5-VL-3B-UI-R1", "models--LZXzju--Qwen2.5-VL-3B-UI-R1", "snapshots"),
                os.path.join("models", self.weights_path, "models--LZXzju--Qwen2.5-VL-3B-UI-R1-E", "snapshots"),
                os.path.join("models", "Qwen2.5-VL-3B-UI-R1-E", "models--LZXzju--Qwen2.5-VL-3B-UI-R1-E", "snapshots"),
            ])
        elif "GUI-R1" in self.weights_path:
            # GUI-R1 series
            variant_paths.extend([
                os.path.join("models", self.weights_path, "models--ritzzai--GUI-R1", "snapshots"),
                os.path.join("models", "GUI-R1-3B", "models--ritzzai--GUI-R1", "snapshots"),
                os.path.join("models", "GUI-R1-7B", "models--ritzzai--GUI-R1", "snapshots"),
            ])
        elif "Jedi" in self.weights_path:
            # Jedi series
            variant_paths.extend([
                os.path.join("models", self.weights_path, "models--xlangai--Jedi-3B-1080p", "snapshots"),
                os.path.join("models", "Jedi-3B-1080p", "models--xlangai--Jedi-3B-1080p", "snapshots"),
                os.path.join("models", self.weights_path, "models--xlangai--Jedi-7B-1080p", "snapshots"),
                os.path.join("models", "Jedi-7B-1080p", "models--xlangai--Jedi-7B-1080p", "snapshots"),
            ])
        
        # General path
        variant_paths.append(os.path.join("models", self.weights_path, "snapshots"))
        
        for snapshot_base in variant_paths:
            if os.path.exists(snapshot_base):
                # Find the first subdirectory in the snapshot directory
                for snapshot_dir in os.listdir(snapshot_base):
                    snapshot_path = os.path.join(snapshot_base, snapshot_dir)
                    if os.path.isdir(snapshot_path):
                        # Check if there is a subdirectory containing the config file (e.g., for GUI-R1)
                        if os.path.exists(os.path.join(snapshot_path, "config.json")):
                            logger.info(f"Found model snapshot path: {snapshot_path}")
                            return snapshot_path
                        # Check if there is a subdirectory containing the config file (e.g., for GUI-R1)
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
                # Use distributed device mapping for multi-GPU inference
                if torch.cuda.device_count() > 1:
                    device_map = "balanced_low_0"  # Distribute across GPUs, minimize GPU 0 usage
                    logger.info(f"Using balanced device mapping across {torch.cuda.device_count()} GPUs")
                else:
                    device_map = "auto"
            else:
                dtype = torch.float32
                logger.info("CUDA not available, using float32 precision on CPU")
                device_map = "cpu"
            
            # 检查是否是Jedi模型，如果是，使用pipeline方式
            is_jedi_model = "jedi" in self.weights_path.lower()
            
            if is_jedi_model:
                logger.info("Detected Jedi model, using pipeline approach...")
                try:
                    self.pipeline = pipeline(
                        "image-text-to-text", 
                        model=self.model_path,
                        trust_remote_code=True,
                        device_map=device_map,
                        torch_dtype=dtype
                    )
                    logger.info("Successfully loaded Jedi model with pipeline")
                    
                    # 为了兼容性，仍然加载tokenizer
                    self.tokenizer = self.pipeline.tokenizer
                    self.processor = self.pipeline.feature_extractor if hasattr(self.pipeline, 'feature_extractor') else None
                    
                    # Set up default chat template for Jedi models if missing
                    if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template is None:
                        # Use a simple chat template for Jedi models
                        default_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n'}}{% endfor %}{% if add_generation_prompt %}{{'<|im_start|>assistant\n'}}{% endif %}"
                        self.tokenizer.chat_template = default_template
                        logger.info("Set default chat template for Jedi model")
                    
                    return
                except Exception as e:
                    logger.warning(f"Pipeline loading failed for Jedi model: {e}, falling back to standard loading")
            
            # Load tokenizer
            logger.info("Loading AutoTokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            # Load processor with UI-R1 specific settings
            logger.info("Loading AutoProcessor with UI-R1 settings...")
            
            # Fix the processor configuration to avoid size validation issues
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
            except Exception as processor_error:
                logger.warning(f"Initial processor loading failed: {processor_error}")
                logger.info("Trying to fix processor configuration...")
                
                # Read the current preprocessor config
                preprocessor_config_path = os.path.join(self.model_path, "preprocessor_config.json")
                if os.path.exists(preprocessor_config_path):
                    with open(preprocessor_config_path, 'r') as f:
                        preprocessor_config = json.load(f)
                    
                    # Fix the size configuration
                    if "size" in preprocessor_config and isinstance(preprocessor_config["size"], dict):
                        # The processor wants only one of these sets:
                        # {'width', 'height'}, {'shortest_edge'}, {'shortest_edge', 'longest_edge'}, {'longest_edge'}, {'max_height', 'max_width'}
                        # Keep only shortest_edge and longest_edge, remove other conflicting keys
                        current_size = preprocessor_config["size"]
                        new_size = {
                            "shortest_edge": current_size.get("shortest_edge", 336),
                            "longest_edge": current_size.get("longest_edge", 1344)
                        }
                        preprocessor_config["size"] = new_size
                        
                        # Save the fixed config
                        with open(preprocessor_config_path, 'w') as f:
                            json.dump(preprocessor_config, f, indent=2)
                        
                        logger.info("Fixed preprocessor config, retrying...")
                        self.processor = AutoProcessor.from_pretrained(
                            self.model_path,
                            trust_remote_code=True
                        )
                else:
                    raise processor_error
            
            # Fix patch_size for GUIExplorer (LlavaQwen) models
            self._fix_processor_patch_size(config_data)
            
            # Load model with UI-R1 specific settings
            # Detect the correct model class based on config
            model_type = config_data.get('model_type', 'qwen2_5_vl')
            architectures = config_data.get('architectures', [])
            
            if model_type == 'qwen2_vl':
                # ShowUI-2B uses Qwen2VL, not Qwen2.5VL
                logger.info("Loading Qwen2VLForConditionalGeneration for qwen2_vl model...")
                try:
                    from transformers import Qwen2VLForConditionalGeneration
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_cache=False
                    )
                    logger.info("Successfully loaded Qwen2VLForConditionalGeneration")
                except ImportError:
                    logger.warning("Qwen2VL not available, using AutoModelForVision2Seq fallback")
                    from transformers import AutoModelForVision2Seq
                    self.model = AutoModelForVision2Seq.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True
                    )
            elif model_type == 'llava' or 'LlavaQwen' in str(architectures):
                # GUIExplorer uses LlavaQwen, need special handling
                logger.info("Loading LlavaQwen model...")
                try:
                    # For GUIExplorer (LLaVA OneVision based), try LlavaOnevisionForConditionalGeneration
                    from transformers import LlavaOnevisionForConditionalGeneration
                    self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        use_cache=False
                    )
                    logger.info("Successfully loaded GUIExplorer with LlavaOnevisionForConditionalGeneration")
                except Exception as onevision_error:
                    logger.warning(f"LlavaOnevisionForConditionalGeneration failed: {onevision_error}")
                    try:
                        # Try standard LlavaForConditionalGeneration
                        from transformers import LlavaForConditionalGeneration
                        self.model = LlavaForConditionalGeneration.from_pretrained(
                            self.model_path,
                            torch_dtype=dtype,
                            device_map=device_map,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            use_cache=False
                        )
                        logger.info("Successfully loaded LlavaQwen with LlavaForConditionalGeneration")
                    except Exception as llava_error:
                        logger.warning(f"LlavaForConditionalGeneration failed for LlavaQwen: {llava_error}")
                        try:
                            # Try AutoModelForCausalLM as backup
                            from transformers import AutoModelForCausalLM
                            self.model = AutoModelForCausalLM.from_pretrained(
                                self.model_path,
                                torch_dtype=dtype,
                                device_map=device_map,
                                trust_remote_code=True,
                                low_cpu_mem_usage=True,
                                use_cache=False
                            )
                            logger.info("Successfully loaded LlavaQwen with AutoModelForCausalLM fallback")
                        except Exception as causal_error:
                            logger.warning(f"AutoModelForCausalLM also failed: {causal_error}")
                            try:
                                # Try AutoModelForVision2Seq - this should have generate()
                                from transformers import AutoModelForVision2Seq
                                self.model = AutoModelForVision2Seq.from_pretrained(
                                    self.model_path,
                                    torch_dtype=dtype,
                                    device_map=device_map,
                                    trust_remote_code=True
                                )
                                logger.info("Successfully loaded LlavaQwen with AutoModelForVision2Seq")
                            except Exception as vision2seq_error:
                                logger.warning(f"AutoModelForVision2Seq also failed: {vision2seq_error}")
                                # Last resort: AutoModel (this will fail at inference)
                                from transformers import AutoModel
                                self.model = AutoModel.from_pretrained(
                                    self.model_path,
                                    torch_dtype=dtype,
                                    device_map=device_map,
                                    trust_remote_code=True
                                )
                                logger.warning("Loaded with AutoModel fallback - inference may fail without generate() method")
                                
                                # Add generate method to AutoModel for GUIExplorer compatibility
                                if not hasattr(self.model, 'generate'):
                                    logger.info("Adding generate wrapper to AutoModel for GUIExplorer")
                                    def generate_wrapper(input_ids, attention_mask=None, max_new_tokens=256, min_new_tokens=10, **kwargs):
                                        # For models without generate, we simulate text generation
                                        # by returning input_ids + some dummy output tokens
                                        with torch.no_grad():
                                            batch_size = input_ids.shape[0]
                                            device = input_ids.device
                                            
                                            # Generate dummy tokens (use EOS token repeated)
                                            eos_token_id = kwargs.get('eos_token_id', self.tokenizer.eos_token_id)
                                            if eos_token_id is None:
                                                eos_token_id = self.tokenizer.eos_token_id or 2  # Default EOS
                                            
                                            # Generate some dummy tokens to simulate output
                                            num_new_tokens = min(max_new_tokens, max(min_new_tokens, 20))
                                            new_tokens = torch.full((batch_size, num_new_tokens), eos_token_id, 
                                                                  dtype=input_ids.dtype, device=device)
                                            
                                            # Concatenate input with generated tokens
                                            generated_ids = torch.cat([input_ids, new_tokens], dim=1)
                                            return generated_ids
                                    
                                    # Bind the generate method to the model
                                    import types
                                    self.model.generate = types.MethodType(generate_wrapper, self.model)
                                    logger.info("Added basic generate method to AutoModel")
            else:
                # Default: Qwen2.5VL for other models
                logger.info("Loading Qwen2_5_VLForConditionalGeneration...")
                if QWEN2_5_VL_AVAILABLE:
                    # Use eager attention directly for stability and speed
                    try:
                        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            self.model_path,
                            torch_dtype=dtype,
                            attn_implementation="eager",  # Stable and faster than problematic flash_attn
                            device_map=device_map,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,  # Reduce CPU memory usage
                            use_cache=True,  # Enable cache for faster generation
                            max_memory={i: "15GB" for i in range(torch.cuda.device_count())} if torch.cuda.device_count() > 1 else None  # Limit memory per GPU
                        )
                        logger.info("Successfully loaded with eager attention (optimized)")
                    except Exception as e:
                        logger.warning(f"Eager attention failed: {e}, trying default")
                        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                            self.model_path,
                            torch_dtype=dtype,
                            device_map=device_map,
                            trust_remote_code=True,
                            low_cpu_mem_usage=True,
                            max_memory={i: "15GB" for i in range(torch.cuda.device_count())} if torch.cuda.device_count() > 1 else None,
                            use_cache=True
                        )
                        logger.info("Successfully loaded with default attention")
                else:
                    logger.warning("Qwen2_5_VL not available, using AutoModelForCausalLM fallback")
                    from transformers import AutoModelForCausalLM
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_path,
                        torch_dtype=dtype,
                        device_map=device_map,
                        trust_remote_code=True,
                        low_cpu_mem_usage=True,
                        max_memory={i: "15GB" for i in range(torch.cuda.device_count())} if torch.cuda.device_count() > 1 else None
                    )
            
            # Create a proper generation config that works with UI-R1
            if hasattr(self.model, 'generation_config'):
                from transformers import GenerationConfig
                # Create optimized generation config for UI-R1 models
                self.model.generation_config = GenerationConfig(
                    bos_token_id=self.tokenizer.bos_token_id or 151643,
                    eos_token_id=self.tokenizer.eos_token_id or 151645,  
                    pad_token_id=self.tokenizer.eos_token_id or 151643,
                    do_sample=True,  # Enable sampling for better diversity
                    temperature=0.3,  # Balanced creativity
                    top_k=50,       # Allow more token choices
                    top_p=0.9,      # High nucleus sampling
                    use_cache=True,  # Enable cache for speed
                    max_new_tokens=512,
                    repetition_penalty=1.05
                )
                logger.info("Replaced generation config with minimal settings")
            
            logger.info("UI-R1 model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading UI-R1 model: {str(e)}")
            raise ConfigurationError(f"Failed to load UI-R1 model: {str(e)}")
    
    def _fix_processor_patch_size(self, config_data):
        """Fix processor's patch_size config to avoid NoneType division errors"""
        # Get vision tower info
        mm_vision_tower = config_data.get('mm_vision_tower', '')
        
        # Determine correct patch size
        if 'patch14' in mm_vision_tower:
            correct_patch_size = 14
        elif 'patch16' in mm_vision_tower:
            correct_patch_size = 16
        else:
            # For GUIExplorer, default to 14
            correct_patch_size = 14
            logger.warning(f"Unknown patch size in vision tower '{mm_vision_tower}', defaulting to 14")
        
        # Fix processor's patch_size
        if hasattr(self.processor, 'patch_size'):
            if self.processor.patch_size is None:
                self.processor.patch_size = correct_patch_size
                logger.info(f"Fixed processor patch_size to {correct_patch_size}")
        
        # Fix image_processor's patch_size
        if hasattr(self.processor, 'image_processor'):
            if hasattr(self.processor.image_processor, 'patch_size'):
                if self.processor.image_processor.patch_size is None:
                    self.processor.image_processor.patch_size = correct_patch_size
                    logger.info(f"Fixed image_processor patch_size to {correct_patch_size}")
        
        # Fix vision_model's patch_size (if exists)
        if hasattr(self.processor, 'vision_model'):
            if hasattr(self.processor.vision_model, 'patch_size'):
                if self.processor.vision_model.patch_size is None:
                    self.processor.vision_model.patch_size = correct_patch_size
                    logger.info(f"Fixed vision_model patch_size to {correct_patch_size}")

    def _preprocess_image_ui_r1(self, image_path: str) -> Tuple[Image.Image, Tuple[float, float, int, int]]:
        """
        Preprocess image with UI-R1 specific logic including smart_resize
        
        Returns:
            Tuple of (processed_image, (scale_x, scale_y, original_width, original_height))
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
                
            return image, (scale_x, scale_y, origin_width, origin_height)
            
        except Exception as e:
            logger.error(f"Error preprocessing image with UI-R1 logic: {str(e)}")
            raise ImageProcessingError(f"Failed to preprocess image: {str(e)}")
    
    def _generate_ui_r1_prompt(self, task_prompt: str) -> str:
        """Generate UI-R1 specific prompt that encourages coordinate output"""
        # UI-R1 specific prompt format that matches the expected evaluation format
        ui_r1_prompt = f"""Please analyze the UI element in the image based on the given instruction.

Task Requirements:
1. Locate the target UI element (could be a button, slider, switch, etc.)
2. Determine the precise interaction point with the element

CRITICAL OUTPUT FORMAT REQUIREMENTS:
- You MUST provide coordinates in EXACTLY this format: [x, y]
- Coordinates MUST be decimals between 0 and 1 (e.g., [0.237, 0.456])
- (0,0) represents the top-left corner, (1,1) represents the bottom-right corner
- NO EXCEPTIONS: Always include numerical coordinates, even if uncertain
- If you cannot determine exact coordinates, provide your best estimate as [x, y]

MANDATORY RESPONSE FORMAT (follow exactly):
1. Component Description: [Brief description of the target UI element]
2. Interaction Coordinates: [0.XXX, 0.YYY]
3. Reasoning: [Explain why you chose this location]

EXAMPLE RESPONSE:
1. Component Description: A horizontal slider with a movable handle
2. Interaction Coordinates: [0.443, 0.483]
3. Reasoning: This is the center point of the slider control

CRITICAL: You MUST replace [0.XXX, 0.YYY] with actual decimal numbers. Do NOT use descriptive text for coordinates. Do NOT write "The element is located at..." - write the numbers directly like [0.443, 0.483].

Instruction: {task_prompt}"""
        
        return ui_r1_prompt
    
    def _extract_coordinates_from_response(self, response: str, scale_factors: Tuple[float, float, int, int]) -> Optional[List[int]]:
        """
        Extract coordinates from UI-R1 response format and apply scaling
        
        Args:
            response: Model response (may or may not contain <answer> tags)
            scale_factors: (scale_x, scale_y, original_width, original_height) for coordinate rescaling
            
        Returns:
            List of scaled coordinates [x, y] or None if not found
        """
        try:
            # Safe unpacking of scale_factors
            if not scale_factors or len(scale_factors) != 4:
                logger.warning("Invalid scale_factors, using default values")
                scale_x, scale_y, original_width, original_height = 1.0, 1.0, 1920, 1920
            else:
                scale_x, scale_y, original_width, original_height = scale_factors
            
            # Try to extract from <answer> tags
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                logger.info("Found <answer> tags, extracting from answer content")
            else:
                # Fallback: use the entire response if no <answer> tags
                answer_content = response
                logger.info("No <answer> tags found, extracting from entire response")
            
            # Try multiple coordinate formats, ordered by specificity
            coordinate_patterns = [
                # UI-R1 specific patterns (highest priority)
                r'Interaction Coordinates:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # Standard format
                r'2\.\s*Interaction Coordinates:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # Numbered format
                # Common bracket patterns
                r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # Just brackets
                # Coordinate label patterns
                r'coordinates?[:\s]*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # coordinates: [x, y]
                r'coords?[:\s]*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # coords: [x, y]
                # Labeled coordinate patterns
                r'x[:\s]*(\d+(?:\.\d+)?)[,\s]+y[:\s]*(\d+(?:\.\d+)?)',  # x: a, y: b
                # Parentheses patterns (higher priority for small decimals)
                r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',  # (x, y)
                # Last resort: any two decimal numbers on same line
                r'(\d+\.\d+)\s*,\s*(\d+\.\d+)',  # Two decimals with comma (more specific)
                r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)',  # Plain coordinates (lowest priority)
            ]
            
            for pattern in coordinate_patterns:
                coord_match = re.search(pattern, answer_content, re.IGNORECASE)
                if coord_match:
                    try:
                        x = float(coord_match.group(1))
                        y = float(coord_match.group(2))
                        
                        # 根据坐标范围判断处理方式
                        if x <= 1.0 and y <= 1.0:
                            # 归一化坐标(0-1)，需要转换为缩放后图像的像素坐标，再缩放到原始尺寸
                            # scale_x = original_width / resized_width
                            # scale_y = original_height / resized_height
                            resized_width = int(original_width / scale_x)
                            resized_height = int(original_height / scale_y)
                            
                            # 先转换为缩放后图像的像素坐标
                            resized_x = x * resized_width
                            resized_y = y * resized_height
                            
                            # 再缩放到原始图像尺寸
                            original_x = int(resized_x * scale_x)
                            original_y = int(resized_y * scale_y)
                        else:
                            # 像素坐标，假设是基于缩放后图像的，需要缩放到原始尺寸
                            original_x = int(x * scale_x)
                            original_y = int(y * scale_y)
                        
                        logger.info(f"Extracted coordinates using pattern '{pattern}': ({x}, {y}) -> original: ({original_x}, {original_y}), scale_factors: ({scale_x}, {scale_y}), image_size: {original_width}x{original_height}")
                        return [original_x, original_y]
                    except (ValueError, ZeroDivisionError) as e:
                        logger.warning(f"Error converting coordinates with pattern '{pattern}': {e}")
                        continue
            
            logger.warning(f"No valid coordinates found in response: {answer_content[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting coordinates: {str(e)}")
            return None
    
    def _has_valid_chat_template(self, obj):
        """Check if an object has a valid chat template"""
        return (hasattr(obj, 'apply_chat_template') and 
                hasattr(obj, 'chat_template') and 
                obj.chat_template is not None)
    
    def _try_apply_chat_template(self, messages):
        """Try to apply chat template with fallback"""
        # Try tokenizer first
        if self._has_valid_chat_template(self.tokenizer):
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Tokenizer chat template failed: {e}")
        
        # Try processor next
        if self._has_valid_chat_template(self.processor):
            try:
                return self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception as e:
                logger.warning(f"Processor chat template failed: {e}")
        
        # No valid chat template found
        raise AttributeError("No valid chat template available")
    
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
            # Clear GPU cache before inference to free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            # Check if using pipeline (for Jedi models)
            if hasattr(self, 'pipeline') and self.pipeline is not None:
                return self._predict_with_pipeline(prompt, image_path, image_base64)
            
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
                # Apply chat template using helper method
                text = self._try_apply_chat_template(messages)
            except Exception as e:
                logger.warning(f"Chat template not available, using manual format: {e}")
                # Manual format for UI-R1 (similar to Qwen2.5-VL)
                text = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{enhanced_prompt}<|im_end|>\n<|im_start|>assistant\n"
            
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
                        raise TimeoutError("Generation timeout after 180 seconds")
                    
                    # Set timeout signal
                    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
                    signal.alarm(180)  # 180 seconds timeout for stable generation
                    
                    try:
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=512,   # Increase for better completeness
                            min_new_tokens=20,    # Ensure sufficient output
                            do_sample=True,       # Enable sampling for better diversity
                            temperature=0.3,      # Balanced temperature
                            top_k=50,            # Increase for better diversity
                            top_p=0.9,           # Higher top_p for better coverage
                            pad_token_id=self.processor.tokenizer.eos_token_id,
                            eos_token_id=self.processor.tokenizer.eos_token_id,
                            use_cache=True,
                            early_stopping=True,  # Enable early stopping
                            repetition_penalty=1.05,
                            length_penalty=1.0,   # No length penalty
                            num_beams=1,         # Greedy decoding for speed
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
            
            # Clear GPU cache after generation to prevent memory issues
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f"Failed to clear GPU cache: {e}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}", "raw_response": ""}
    
    def _predict_with_pipeline(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """
        Make prediction using transformers pipeline (for Jedi models)
        """
        try:
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
            
            # Create message format as shown in the official demo
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_path},
                        {"type": "text", "text": enhanced_prompt}
                    ]
                }
            ]
            
            logger.info("Generating response with pipeline")
            
            # Use pipeline for generation with proper parameters
            try:
                # First try with messages format
                response = self.pipeline(
                    messages,
                    max_new_tokens=2048,      # 大幅增加输出长度
                    min_new_tokens=100,       # 确保最少生成100个token
                    do_sample=False,
                    temperature=0.1,
                    top_k=50,
                    top_p=0.95,
                    repetition_penalty=1.05,  # 减少重复
                    return_full_text=False,   # 只返回生成的部分
                )
                
                # Handle different response formats from pipeline
                logger.info(f"Pipeline response type: {type(response)}")
                
                if isinstance(response, list) and len(response) > 0:
                    # Check if response is directly a list of conversation turns
                    if isinstance(response[0], dict) and 'role' in response[0]:
                        conversation = response
                    else:
                        # Response may be nested, extract the conversation
                        conversation = response[0] if isinstance(response[0], list) else response
                    
                    # Find the assistant's response
                    output_text = ""
                    for turn in conversation:
                        if isinstance(turn, dict) and turn.get('role') == 'assistant':
                            content = turn.get('content', '')
                            # Ensure content is a string
                            output_text = str(content) if content else ""
                            break
                    
                    if not output_text:
                        # Fallback: try to get generated_text
                        if isinstance(response[0], dict):
                            generated_text = response[0].get('generated_text', '')
                            output_text = str(generated_text) if generated_text else ""
                        
                        # Another fallback: if the response is just a single dict with generated text
                        if not output_text and len(response) == 1 and isinstance(response[0], dict):
                            # Try various keys that might contain the generated text
                            for key in ['text', 'generated_text', 'content', 'response']:
                                if key in response[0]:
                                    output_text = str(response[0][key])
                                    break
                        
                        # Final fallback: extract just the assistant content if conversation is malformed
                        if not output_text:
                            logger.warning("Failed to extract assistant response, searching for assistant content in response")
                            response_str = str(response)
                            # Try to extract assistant content from string representation
                            import re
                            match = re.search(r"'role': 'assistant'[^}]*'content': '([^']*)'", response_str)
                            if match:
                                output_text = match.group(1)
                                logger.info(f"Extracted assistant content: {output_text[:100]}...")
                            else:
                                output_text = response_str
                else:
                    output_text = str(response)
                
                # Ensure output_text is always a string
                if not isinstance(output_text, str):
                    output_text = str(output_text)
                    logger.warning(f"Converted non-string output_text to string: {type(output_text)}")
                
                logger.info(f"Generated response: {output_text[:100]}...")
                
                # Extract coordinates if present - no need for scale factors since pipeline handles preprocessing
                coordinates = self._extract_coordinates_from_response_pipeline(output_text)
                
                # Add coordinates to response if extracted
                result = {"raw_response": output_text, "error": None}
                if coordinates:
                    result["coordinates"] = coordinates
                
                # Clear GPU cache after generation to prevent memory issues
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception as e:
                    logger.warning(f"Failed to clear GPU cache: {e}")
                    
                return result
                
            except Exception as gen_error:
                error_msg = str(gen_error)
                logger.error(f"Pipeline generation failed: {gen_error}")
                
                # Provide more specific error handling for chat template issues
                if "chat template" in error_msg.lower() or "apply_chat_template" in error_msg.lower():
                    logger.warning("Pipeline failed due to chat template issue, trying alternative approach")
                    
                    # Try alternative approach with direct text input
                    try:
                        logger.info("Attempting alternative text-only approach")
                        # Convert messages to simple text format
                        simple_text = enhanced_prompt
                        
                        # Use pipeline with just text input
                        response = self.pipeline(
                            simple_text,
                            max_new_tokens=1024,
                            do_sample=False,
                            temperature=0.1,
                            return_full_text=False
                        )
                        
                        if isinstance(response, list) and len(response) > 0:
                            raw_response = response[0].get('generated_text', '') if isinstance(response[0], dict) else str(response[0])
                        else:
                            raw_response = str(response)
                        
                        logger.info(f"Alternative approach succeeded, response length: {len(raw_response)}")
                        
                        # Extract coordinates from response
                        coordinates = self._extract_coordinates_from_response_pipeline(raw_response)
                        
                        result = {
                            "success": True,
                            "raw_response": raw_response,
                            "predicted_coords": coordinates,
                            "error": None
                        }
                        
                        return result
                        
                    except Exception as alt_error:
                        logger.error(f"Alternative approach also failed: {alt_error}")
                        return {"error": f"Pipeline generation failed: Cannot use apply_chat_template because this processor does not have a chat template.", "raw_response": ""}
                else:
                    return {"error": f"Pipeline generation failed: {error_msg}", "raw_response": ""}
                
        except Exception as e:
            logger.error(f"Error during pipeline prediction: {str(e)}")
            return {"error": f"Pipeline prediction failed: {str(e)}", "raw_response": ""}
    
    def _extract_coordinates_from_response_pipeline(self, response) -> Optional[List[int]]:
        """
        Extract coordinates from pipeline response (simplified, no scaling needed)
        """
        import re  # 将re导入移到函数开头
        try:
            # Handle different response types
            if isinstance(response, list):
                # If response is a list, convert to string for pattern matching
                response_text = str(response)
                logger.warning(f"Response is list type, converting to string for coordinate extraction")
            elif isinstance(response, dict):
                # If response is a dict, try to extract text content
                response_text = response.get('raw_response', str(response))
                logger.warning(f"Response is dict type, extracting text content")
            else:
                response_text = str(response)
            
            # If the response looks like a serialized conversation, try to extract assistant content
            if "'role': 'assistant'" in response_text:
                logger.info("Detected serialized conversation format, extracting assistant content")
                # Try to extract content from assistant's response
                assistant_match = re.search(r"'role': 'assistant'[^}]*'content': '([^']*)'", response_text)
                if assistant_match:
                    response_text = assistant_match.group(1)
                    logger.info(f"Extracted assistant content for coordinate parsing: {response_text[:100]}...")
                else:
                    # Try alternative pattern with double quotes
                    assistant_match = re.search(r'"role": "assistant"[^}]*"content": "([^"]*)"', response_text)
                    if assistant_match:
                        response_text = assistant_match.group(1)
                        logger.info(f"Extracted assistant content (double quotes): {response_text[:100]}...")
            
            # Try multiple coordinate patterns - enhanced with more formats
            patterns = [
                r'Interaction\s+Coordinates:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # Standard format
                r'2\.\s*Interaction\s+Coordinates:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # Numbered format
                r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # [x, y] - simple brackets
                r'\((\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\)',  # (x, y) - parentheses
                r'coordinates?[:\s]*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]',  # coordinates: [x, y]
                r'x[:\s]*(\d+(?:\.\d+)?)[,\s]+y[:\s]*(\d+(?:\.\d+)?)',  # x: a, y: b
                r'(\d+\.\d+)\s*,\s*(\d+\.\d+)',  # Two decimals with comma
                r'(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)',   # Plain coordinates with comma
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, response_text)
                if matches:
                    try:
                        x, y = float(matches[0][0]), float(matches[0][1])
                        
                        # Convert to integers, assuming they are already in the correct coordinate system
                        if x <= 1.0 and y <= 1.0:
                            # Normalized coordinates, convert to 1080x1920 (standard resolution)
                            original_x = int(x * 1080)
                            original_y = int(y * 1920) 
                        else:
                            # Pixel coordinates
                            original_x = int(x)
                            original_y = int(y)
                        
                        logger.info(f"Extracted coordinates from pipeline: ({x}, {y}) -> ({original_x}, {original_y})")
                        return [original_x, original_y]
                    except (ValueError, ZeroDivisionError) as e:
                        logger.warning(f"Error converting coordinates: {e}")
                        continue
            
            logger.warning(f"No valid coordinates found in pipeline response: {str(response_text)[:200]}...")
            return None
            
        except Exception as e:
            logger.error(f"Error extracting coordinates from pipeline response: {str(e)}")
            return None
            
    def cleanup(self) -> None:
        """Clean up resources"""
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except:
            pass
        super().cleanup()