import torch
import torch.nn as nn
from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor
from PIL import Image
import json
import os
from typing import Dict, List, Optional, Tuple

class MobileVLMModel:
    
    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = model_path
        self.device = device
        self.tokenizer = None
        self.model = None
        self.vision_tower = None
        self.image_processor = None
        self.mm_projector = None
        
        with open(os.path.join(model_path, "config.json"), "r") as f:
            self.config = json.load(f)
        
        self._load_tokenizer()
        self._load_vision_tower()
        self._load_language_model()
        self._setup_mm_projector()
    
    def _load_tokenizer(self):
        self.tokenizer = LlamaTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            use_fast=False
        )
        
        # Add special tokens for image
        if "<image>" not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens(["<image>"], special_tokens=True)
    
    def _load_vision_tower(self):
        vision_tower_name = self.config.get("mm_vision_tower", "openai/clip-vit-large-patch14-336")
        
        try:
            import transformers
            transformers.utils.hub.HUGGINGFACE_CO_RESOLVE_ENDPOINT = None
            
            self.vision_tower = CLIPVisionModel.from_pretrained(
                vision_tower_name,
                torch_dtype=torch.bfloat16,
                local_files_only=True
            ).to(self.device)
            
            self.image_processor = CLIPImageProcessor.from_pretrained(
                vision_tower_name,
                local_files_only=True
            )
            print(f"Vision tower loaded from cache: {vision_tower_name}")
        except Exception as e:
            print(f"Failed to load vision tower from cache: {e}")

            self.image_processor = CLIPImageProcessor(
                size=336,
                crop_size=336,
                do_normalize=True,
                image_mean=[0.48145466, 0.4578275, 0.40821073],
                image_std=[0.26862954, 0.26130258, 0.27577711]
            )
            
            self.vision_tower = None
            print("Running in text-only mode (no vision tower)")
    
    def _load_language_model(self):
        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        if len(self.tokenizer) > self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))
    
    def _setup_mm_projector(self):
        mm_hidden_size = self.config.get("mm_hidden_size", 1024)
        hidden_size = self.config.get("hidden_size", 4096)

        self.mm_projector = nn.Linear(mm_hidden_size, hidden_size).to(self.device)

        with torch.no_grad():
            self.mm_projector.weight.normal_(mean=0.0, std=0.02)
            self.mm_projector.bias.zero_()
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to features"""
        if self.vision_tower is None:
            hidden_size = self.config.get("hidden_size", 4096)
            image_features = torch.zeros(1, hidden_size, dtype=torch.bfloat16).to(self.device)
            return image_features

        image_inputs = self.image_processor(
            images=image,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            image_features = self.vision_tower(**image_inputs).last_hidden_state

        image_features = image_features[:, 0, :]  
        
        image_features = self.mm_projector(image_features)
        
        return image_features
    
    def prepare_inputs(self, prompt: str, image: Image.Image) -> Dict:

        image_features = self.encode_image(image)
        
        if "<image>" not in prompt:
            prompt = f"<image>\n{prompt}"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        inputs["image_features"] = image_features
        
        return inputs
    
    def generate(self, prompt: str, image: Image.Image, max_new_tokens: int = 512) -> str:

        inputs = self.prepare_inputs(prompt, image)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        if prompt_text in response:
            response = response.replace(prompt_text, "").strip()
        
        return response
    
    def cleanup(self):
        """Clean up resources"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    model_path = "/home/yjp/spiderbench/models/MobileVLM_V2-7B"
    test_image = "/home/yjp/spiderbench/element_detection/1637.jpg"
    
    if os.path.exists(model_path) and os.path.exists(test_image):
        print("Testing MobileVLM model...")
        
        model = MobileVLMModel(model_path)
        
        image = Image.open(test_image).convert("RGB")
        prompt = "Describe what you see in this image."
        
        response = model.generate(prompt, image)
        print(f"Response: {response}")
        
        model.cleanup()
    else:
        print("Model or test image not found")