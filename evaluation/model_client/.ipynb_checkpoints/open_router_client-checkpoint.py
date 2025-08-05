#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
import requests
import json
from typing import Dict, List

from ..model_clients import ModelClient, ConfigurationError, APIError, ImageProcessingError

logger = logging.getLogger("OpenRouterClient")

class OpenRouterClient(ModelClient):
    """OpenRouter API client implementation"""
    
    def __init__(self, model_name: str, api_key: str = None, config: Dict = None, use_component_detector: bool = False):
        """
        Initialize OpenRouter client
        
        Args:
            model_name: Name of the model
            api_key: API key for OpenRouter
            config: Configuration dictionary
            use_component_detector: Whether to use component detection
        """
        super().__init__(model_name, api_key, config, use_component_detector)
        
        # Initialize API configuration
        self.api_url = self.config.get("api_url", "https://openrouter.ai/api/v1/chat/completions")
        self.extra_headers = self.config.get("extra_headers", {})
        
        # Set up HTTP session
        self.session = self._setup_session()
        
    def _validate_model_specific_config(self) -> None:
        """Validate OpenRouter specific configuration"""
        if not self.api_key:
            raise ConfigurationError("API key is required for OpenRouter client")
            
    def _setup_session(self) -> requests.Session:
        """Set up HTTP session with retry and pooling"""
        session = requests.Session()
        session.verify = True
        session.mount('https://', requests.adapters.HTTPAdapter(
            max_retries=3,
            pool_connections=10,
            pool_maxsize=10
        ))
        return session
        
    def _prepare_messages(self, prompt: str, image_data: str) -> List[Dict]:
        """
        Prepare messages for the API request
        
        Args:
            prompt: Text prompt
            image_data: Base64 encoded image data
            
        Returns:
            List[Dict]: Formatted messages
        """
        messages = []
        if image_data:
            # Use model-specific image format
            if "qwen" in self.model_name.lower():
                content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            else:
                content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "user", "content": prompt})
            
        return messages
        
    def _prepare_headers(self) -> Dict[str, str]:
        """
        Prepare request headers
        
        Returns:
            Dict[str, str]: Request headers
        """
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": self.extra_headers.get("HTTP-Referer", "https://openrouter.ai/"),
            "X-Title": self.extra_headers.get("X-Title", "Slider UI Evaluation")
        }
        
    def predict(self, prompt: str, image_path: str, image_base64: str = "") -> Dict:
        """
        Send prediction request to OpenRouter API
        
        Args:
            prompt: Text prompt
            image_path: Path to image file
            image_base64: Optional pre-encoded image
            
        Returns:
            Dict: Prediction results
            
        Raises:
            APIError: If API request fails
            ImageProcessingError: If image processing fails
        """
        try:
            # Prepare image data
            image_data = self._prepare_image(image_path, image_base64)
            if not image_data and image_path:
                logger.warning(f"Failed to load image: {image_path}")
                return {"error": "Failed to load image", "raw_response": ""}
                
            logger.info(f"Successfully loaded image: {image_path}, size: {len(image_data) // 1024}KB")
            
            # Prepare prompt
            enhanced_prompt = self.prepare_prompt(prompt, image_path)
            logger.info(f"Processed prompt: {enhanced_prompt[:100]}...")
            
            # Prepare request
            messages = self._prepare_messages(enhanced_prompt, image_data)
            headers = self._prepare_headers()
            
            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 1000,
                "stream": False
            }
            
            # Log request details (excluding sensitive data)
            safe_payload = payload.copy()
            if "messages" in safe_payload:
                safe_payload["messages"] = "[REDACTED]"
            logger.info(f"Sending request to: {self.api_url}")
            logger.info(f"Request parameters: {json.dumps(safe_payload, ensure_ascii=False)}")
            
            # Send request
            try:
                response = self.session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60
                )
            except requests.exceptions.RequestException as e:
                raise APIError(f"API request failed: {str(e)}")
                
            # Check response status
            if response.status_code != 200:
                error_msg = f"OpenRouter API error: status={response.status_code}"
                try:
                    error_detail = response.json()
                    error_msg += f", details: {json.dumps(error_detail, ensure_ascii=False)}"
                except:
                    error_msg += f", response: {response.text[:500]}"
                raise APIError(error_msg)
                
            # Parse response
            try:
                result = response.json()
                logger.info("Successfully parsed API response")
                response_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"Got response text, length: {len(response_text)}")
                return {"raw_response": response_text}
            except json.JSONDecodeError as e:
                raise APIError(f"Failed to parse response JSON: {str(e)}, response: {response.text[:500]}")
                
        except (APIError, ImageProcessingError) as e:
            # Re-raise known errors
            raise
        except Exception as e:
            # Wrap unknown errors
            raise APIError(f"Unexpected error during prediction: {str(e)}")
            
    def cleanup(self):
        """Clean up resources"""
        if hasattr(self, 'session') and self.session:
            try:
                self.session.close()
            except:
                pass
        super().cleanup() 