import os
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

logger = logging.getLogger("PromptCache")

class PromptCache:
    
    def __init__(self, cache_source_dir: str, model_name: str, scenario: int):
        self.cache_source_dir = Path(cache_source_dir)
        self.model_name = model_name
        self.scenario = scenario
        self.model_specific_cache = {}  
        self.image_to_prompts = {}  
        self.loaded = False
        self.logger = logging.getLogger("PromptCache")
        
    def load_cache(self) -> bool:

        try:
            if not self.cache_source_dir.exists():
                self.logger.error(f"Cache source directory does not exist: {self.cache_source_dir}")
                return False

            result_files = list(self.cache_source_dir.glob("result_*.json"))
            if not result_files:
                self.logger.error(f"Result files not found: {self.cache_source_dir}")
                return False
                
            success_count = 0
            for file_path in result_files:
                if self._load_single_result_file(file_path):
                    success_count += 1
                    
            self.loaded = success_count > 0
            if self.loaded:
                self.logger.info(f"Successfully loaded {success_count} prompt files")
                self.logger.info(f"Model-specific cache statistics: {self.get_cache_stats()}")
            else:
                self.logger.error("Failed to successfully load any prompt files")
                
            return self.loaded
            
        except Exception as e:
            self.logger.error(f"Failed to load cache: {str(e)}")
            return False
            
    def _load_single_result_file(self, file_path: Path) -> bool:

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            image_path = data.get("image_path", "")
            if not image_path:
                return False

            normalized_path = self._normalize_image_path(image_path)

            if self.scenario == 1:
                return False
            elif self.scenario == 2:
                prompts = {
                    "original_locate_prompt": data.get("original_locate_prompt", ""),
                    "original_interact_prompt": data.get("original_interact_prompt", ""),
                    "enhanced_locate_prompt": data.get("enhanced_locate_prompt", ""),  
                    "enhanced_interact_prompt": data.get("enhanced_interact_prompt", "")  
                }
            else:  

                prompts = {
                    "original_locate_prompt": data.get("original_locate_prompt", ""),
                    "original_interact_prompt": data.get("original_interact_prompt", ""),
                    "enhanced_locate_prompt": data.get("enhanced_locate_prompt", ""),  
                    "enhanced_interact_prompt": data.get("enhanced_interact_prompt", "")  
                }
           
            if self.scenario in [2, 3]:
                enhanced_locate = prompts.get("enhanced_locate_prompt", "")
                enhanced_interact = prompts.get("enhanced_interact_prompt", "")
                
                if not enhanced_locate and not enhanced_interact:
                    self.logger.warning(f"Valid enhanced prompt not found in file {file_path.name}")
                    return False

                if enhanced_locate and "COMPONENT INFORMATION" not in enhanced_locate:
                    self.logger.warning(f"The enhanced_locate_prompt in file {file_path.name} does not seem to contain component information")
                if enhanced_interact and "COMPONENT INFORMATION" not in enhanced_interact:
                    self.logger.warning(f"The enhanced_interact_prompt in file {file_path.name} does not seem to contain component information")

            if self.model_name not in self.model_specific_cache:
                self.model_specific_cache[self.model_name] = {}
            self.model_specific_cache[self.model_name][normalized_path] = prompts

            self.image_to_prompts[normalized_path] = prompts
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load file {file_path}: {str(e)}")
            return False
            
    def get_cached_prompt(self, image_path: str, prompt_type: str) -> Optional[str]:

        if not self.loaded:
            self.logger.warning("Cache not yet loaded, please call load_cache() first")
            return None

        normalized_path = self._normalize_image_path(image_path)

        if self.scenario == 1:
            return None
        elif self.scenario == 2:
            if self.model_name in self.model_specific_cache:
                model_cache = self.model_specific_cache[self.model_name]
                if normalized_path in model_cache:
                    return model_cache[normalized_path].get(prompt_type)
        else:  
            if self.model_name in self.model_specific_cache:
                model_cache = self.model_specific_cache[self.model_name]
                if normalized_path in model_cache:
                    return model_cache[normalized_path].get(prompt_type)

            if normalized_path in self.image_to_prompts:
                return self.image_to_prompts[normalized_path].get(prompt_type)

            image_filename = os.path.basename(normalized_path)
            for cached_path, prompts in self.image_to_prompts.items():
                if os.path.basename(cached_path) == image_filename:
                    return prompts.get(prompt_type)
        
        return None
        
    def _normalize_image_path(self, image_path: str) -> str:

        if not image_path:
            return ""

        if image_path.startswith("element_detection/"):
            return image_path
        elif "/element_detection/" in image_path:
            return image_path.split("/element_detection/")[-1]
        else:
            filename = os.path.basename(image_path)
            return f"element_detection/{filename}"
        
    def get_cache_stats(self) -> Dict[str, Any]:

        return {
            "total_prompts": len(self.image_to_prompts),
            "model_specific_prompts": len(self.model_specific_cache.get(self.model_name, {})),
            "loaded": self.loaded,
            "source_dir": str(self.cache_source_dir),
            "model_name": self.model_name,
            "scenario": self.scenario
        }


class CachedModelClient:
    
    def __init__(self, original_client, prompt_cache: PromptCache, model_name: str, scenario: int):

        self.original_client = original_client
        self.prompt_cache = prompt_cache
        self.model_name = model_name
        self.scenario = scenario
        self.prompt_hits = 0
        self.prompt_misses = 0
        self.logger = logging.getLogger("CachedModelClient")
        
    def get_response(self, image_path: str, prompt: str) -> str:

        try:
            enhanced_prompt = self._get_enhanced_prompt(image_path, prompt)
            
            if enhanced_prompt:
                self.prompt_hits += 1
                self.logger.info(f"Using enhanced prompt (hit rate: {self.get_prompt_hit_rate():.2%})")
                self.logger.debug(f"Original prompt: {prompt}")
                self.logger.debug(f"Enhanced prompt: {enhanced_prompt}")
            else:
                self.prompt_misses += 1
                enhanced_prompt = prompt
                self.logger.info(f"Using original prompt (hit rate: {self.get_prompt_hit_rate():.2%})")
                
            response = self.original_client.get_response(image_path, enhanced_prompt)

            if isinstance(response, dict):
                text_response = response.get('raw_response', '')
                if not text_response or (isinstance(text_response, str) and text_response.strip() == ""):
                    self.logger.error("Model returned empty response")
                    return "Error: Empty response from model"
                self.logger.debug(f"Model response (dict): {response}")
                return text_response
            elif isinstance(response, str):
                if not response or response.strip() == "":
                    self.logger.error("Model returned empty response")
                    return "Error: Empty response from model"
                self.logger.debug(f"Model response (str): {response}")
                return response
            else:
                text_response = str(response)
                if not text_response or text_response.strip() == "":
                    self.logger.error("Model returned empty response")
                    return "Error: Empty response from model"
                self.logger.debug(f"Model response (other): {response}")
                return text_response
            
        except Exception as e:
            self.logger.error(f"An error occurred while getting response: {str(e)}")
            return f"Error: {str(e)}"
            
    def _get_enhanced_prompt(self, image_path: str, prompt: str) -> Optional[str]:

        try:
            if "COMPONENT INFORMATION" in prompt:
                return None

            if "LOCATE" in prompt.upper():
                return self.prompt_cache.get_cached_prompt(image_path, "enhanced_locate_prompt")
            elif "INTERACT" in prompt.upper():
                return self.prompt_cache.get_cached_prompt(image_path, "enhanced_interact_prompt")
            else:
                return None
        except Exception as e:
            self.logger.error(f"An error occurred while getting enhanced prompt: {str(e)}")
            return None
            
    def get_prompt_hit_rate(self) -> float:

        total = self.prompt_hits + self.prompt_misses
        return self.prompt_hits / total if total > 0 else 0.0
        
    def get_cache_stats(self) -> Dict[str, Any]:

        return {
            "prompt_hits": self.prompt_hits,
            "prompt_misses": self.prompt_misses,
            "hit_rate": self.get_prompt_hit_rate(),
            "total_requests": self.prompt_hits + self.prompt_misses,
            "model_name": self.model_name,
            "scenario": self.scenario,
            "timestamp": datetime.now().isoformat()
        }
        
    def __getattr__(self, name):

        return getattr(self.original_client, name)


def create_cached_model_client(original_client, cache_source_dir: str, model_name: str, scenario: int):

    if not cache_source_dir or not os.path.exists(cache_source_dir):
        logger.warning(f"Cache source directory does not exist or not specified: {cache_source_dir}")
        logger.warning("Using original model client, prompt enhancement not used")
        return original_client
    

    prompt_cache = PromptCache(cache_source_dir, model_name, scenario)
    
    if prompt_cache.load_cache():
        logger.info(f"Successfully created model client with prompt enhancement, using cache source: {cache_source_dir}")
        logger.info(f"Cache stats: {prompt_cache.get_cache_stats()}")
        return CachedModelClient(original_client, prompt_cache, model_name, scenario)
    else:
        logger.warning(f"Failed to load cache from {cache_source_dir}, using original model client")
        return original_client