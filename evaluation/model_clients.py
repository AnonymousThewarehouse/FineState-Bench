#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import base64
import logging
import time
from typing import Dict, Optional, List, Union, Any
import requests
import hashlib
from PIL import Image
import torch
from abc import ABC, abstractmethod
from pathlib import Path
import io
import sys

try:
    from transformers import (
        AutoTokenizer, AutoProcessor, AutoModelForCausalLM, 
        PreTrainedModel, PreTrainedTokenizer, AutoConfig, AutoModel,
        AutoModelForVision2Seq
    )
except ImportError:
    print("Warning: transformers library import failed, please ensure it is installed")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_eval.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ModelClients")

TORCH_AVAILABLE = torch.cuda.is_available()

from .utils import image_to_base64
from .Plug_and_play_model import ComponentDetector

class ConfigurationError(Exception):
    pass

class ModelClientError(Exception):
    pass

class ImageProcessingError(ModelClientError):
    pass

class APIError(ModelClientError):
    pass

class ModelClient(ABC):

    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        self.model_name = model_name
        self.api_key = api_key
        self.config = config or {}
        self.use_component_detector = use_component_detector

        self._validate_config()

        if use_component_detector:
            self._init_component_detector()

        self.logger = logging.getLogger(f"{self.__class__.__name__}_{model_name}")
        
    def _validate_config(self) -> None:
        if not self.model_name:
            raise ConfigurationError("Model name is required")

        self._validate_model_specific_config()
        
    @abstractmethod
    def _validate_model_specific_config(self) -> None:
        pass
        
    def _init_component_detector(self) -> None:
        try:
            use_ground_truth = self.config.get("use_ground_truth", True)
            self.component_detector = ComponentDetector(api_key=self.api_key)
            self.component_detector.set_use_ground_truth(use_ground_truth)
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize component detector: {str(e)}")
            
    def _prepare_image(self, image_path: str, image_base64: str = "") -> str:

        try:
            if image_base64:
                return image_base64
            
            if not image_path:
                return ""

            possible_paths = [
                image_path,
                os.path.join("element_detection", os.path.basename(image_path))
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    self.logger.info(f"Found image file: {path}")
                    try:
                        return image_to_base64(path) or ""
                    except Exception as e:
                        self.logger.warning(f"Failed to encode image {path}: {str(e)}")
                        continue
                        
            raise ImageProcessingError(f"Image not found in any of the locations: {possible_paths}")
            
        except Exception as e:
            raise ImageProcessingError(f"Failed to prepare image: {str(e)}")
            
    def prepare_prompt(self, prompt: str, image_path: str) -> str:

        enhanced_prompt = prompt
        
        if self.use_component_detector and self.component_detector and image_path:
            try:
                if not hasattr(self, 'weights_path') and not self.api_key and not self.component_detector.use_ground_truth:
                    self.logger.error("API key is empty, cannot use component detector")
                    return prompt

                self.component_detector.api_key = self.api_key

                detection_result = self.component_detector.detect_component(image_path)
                
                if detection_result.get("success", False):
                    component_info = {
                        "description": detection_result.get("description", ""),
                        "operation": detection_result.get("operation", ""),
                        "bbox": detection_result.get("bbox")
                    }
                    
                    from .LLM_eval import enhance_prompt_with_component_info
                    enhanced_prompt = enhance_prompt_with_component_info(prompt, component_info)
                    self.logger.info("Successfully enhanced prompt with component information")
                else:
                    self.logger.warning(f"Component detection failed: {detection_result.get('error', 'Unknown error')}")
            except Exception as e:
                self.logger.error(f"Error enhancing prompt: {str(e)}")
                
        return enhanced_prompt
        
    @abstractmethod
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        pass
        
    def get_response(self, image_path: str, prompt: str) -> str:

        try:
            result = self.predict(prompt, image_path)
            if "error" in result and result['error'] is not None:
                return f"Error: {result['error']}"
            return result.get("raw_response", "")
        except Exception as e:
            self.logger.error(f"Error getting response: {str(e)}")
            return f"Error: {str(e)}"
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        pass

def get_model_client(model_name: str, api_key: str, config: Dict, use_component_detector: bool = False) -> ModelClient:

    model_type = config.get("model_type", "online")
    logger = logging.getLogger("ModelClientFactory")
    
    if model_type == "offline":

        if config.get("status") == "incompatible":
            raise ConfigurationError(f"Model {model_name} is marked as incompatible: {config.get('description', 'Unknown issue')}")

        if TORCH_AVAILABLE:
            try:

                client_type = config.get("client_type", "").lower()
                model_name_lower = model_name.lower()

                if client_type == "ui_r1":
                    try:
                        from .model_client import get_ui_r1_client
                        UIR1Client = get_ui_r1_client()
                        logger.info(f"Using UI-R1 client (config-specified) for: {model_name}")
                        return UIR1Client(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import UIR1Client: {e}")
                elif client_type == "chatglm":
                    try:
                        from .model_client import get_chatglm_client
                        ChatGLMClient = get_chatglm_client()
                        logger.info(f"Using ChatGLM client (config-specified) for: {model_name}")
                        return ChatGLMClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import ChatGLMClient: {e}")
                elif client_type == "guiexplorer":
                    try:
                        from .model_client.guiexplorer_client import GUIExplorerClient
                        logger.info(f"Using GUIExplorer client (config-specified) for: {model_name}")
                        return GUIExplorerClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import GUIExplorerClient: {e}")
                

                if "os-atlas" in model_name_lower:
                    try:

                        from .model_client import get_os_atlas_client
                        OSAtlasClient = get_os_atlas_client()
                        logger.info(f"Using specialized OS-Atlas client for: {model_name}")
                        return OSAtlasClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import OSAtlasClient, falling back to OfflineModelClient: {e}")

                if "infigui" in model_name_lower:
                    try:

                        from .model_client import get_infigui_client
                        InfiGUIClient = get_infigui_client()
                        logger.info(f"Using specialized InfiGUI client for: {model_name}")
                        return InfiGUIClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import InfiGUIClient, falling back to OfflineModelClient: {e}")

                if "guiexplorer" in model_name_lower:
                    try:
                        from .model_client.guiexplorer_client import GUIExplorerClient
                        logger.info(f"Using specialized GUIExplorer client for: {model_name}")
                        return GUIExplorerClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import GUIExplorerClient, falling back to OfflineModelClient: {e}")

                if any(variant in model_name_lower for variant in ["ui-r1", "gui-r1", "jedi", "holo1", "showui", "ui-tars"]):
                    try:

                        from .model_client import get_ui_r1_client
                        UIR1Client = get_ui_r1_client()
                        logger.info(f"Using specialized UI-R1 client for: {model_name}")
                        return UIR1Client(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import UIR1Client, falling back to OfflineModelClient: {e}")

                if "agentcpm" in model_name_lower or "minicpmv" in model_name_lower:
                    try:

                        from .model_client import get_minicpmv_client
                        MiniCPMVClient = get_minicpmv_client()
                        logger.info(f"Using specialized MiniCPMV client for: {model_name}")
                        return MiniCPMVClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import MiniCPMVClient, falling back to OfflineModelClient: {e}")

                if "cogagent" in model_name_lower or "chatglm" in model_name_lower:
                    try:

                        from .model_client import get_chatglm_client
                        ChatGLMClient = get_chatglm_client()
                        logger.info(f"Using specialized ChatGLM client for: {model_name}")
                        return ChatGLMClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import ChatGLMClient, falling back to OfflineModelClient: {e}")

                if "mobilevlm" in model_name_lower or "mobilellama" in model_name_lower:
                    try:

                        from .model_client import get_mobilellama_client
                        MobileLlamaClient = get_mobilellama_client()
                        logger.info(f"Using specialized MobileLlama client for: {model_name}")
                        return MobileLlamaClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import MobileLlamaClient, falling back to OfflineModelClient: {e}")

                if "web-llama2" in model_name_lower or "adapter" in model_name_lower:
                    try:

                        from .model_client import get_lora_adapter_client
                        LoRAAdapterClient = get_lora_adapter_client()
                        logger.info(f"Using specialized LoRA adapter client for: {model_name}")
                        return LoRAAdapterClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import LoRAAdapterClient, falling back to OfflineModelClient: {e}")

                if "spiritsight" in model_name_lower:
                    try:

                        from .model_client.spiritsight_client import SpiritSightClient
                        logger.info(f"Using specialized SpiritSight client for: {model_name}")
                        return SpiritSightClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import SpiritSightClient, falling back to OfflineModelClient: {e}")

                if "internvl" in model_name_lower:
                    try:

                        from .model_client.internvl_client import InternVLClient
                        logger.info(f"Using specialized InternVL client for: {model_name}")
                        return InternVLClient(model_name, api_key, config, use_component_detector=use_component_detector)
                    except ImportError as e:
                        logger.warning(f"Failed to import InternVLClient, falling back to OfflineModelClient: {e}")


                logger.info(f"Using offline model client for: {model_name}")
                from .model_client import get_offline_model_client
                OfflineModelClient = get_offline_model_client()
                return OfflineModelClient(model_name, api_key, config, use_component_detector=use_component_detector)
            except Exception as e:
                raise ConfigurationError(f"Failed to initialize offline model client: {str(e)}")
        else:
            logger.warning(f"PyTorch not available, cannot use offline model: {model_name}")
            raise ImportError("PyTorch is required for offline models but not installed")
    else:
        logger.info(f"Using OpenRouter client for: {model_name}")
        from .model_client import get_open_router_client
        OpenRouterClient = get_open_router_client()
        return OpenRouterClient(model_name, api_key, config, use_component_detector=use_component_detector)