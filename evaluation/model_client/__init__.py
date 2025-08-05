#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Model client module initialization file
"""

# Use lazy import to avoid circular import issues
def get_offline_model_client():
    from .offline_model_client import OfflineModelClient
    return OfflineModelClient

def get_open_router_client():
    from .open_router_client import OpenRouterClient
    return OpenRouterClient

def get_os_atlas_client():
    from .os_atlas_client import OSAtlasClient
    return OSAtlasClient

def get_infigui_client():
    from .infigui_client import InfiGUIClient
    return InfiGUIClient

def get_ui_r1_client():
    from .ui_r1_client import UIR1Client
    return UIR1Client

def get_minicpmv_client():
    from .minicpmv_client import MiniCPMVClient
    return MiniCPMVClient

def get_chatglm_client():
    from .chatglm_client import ChatGLMClient
    return ChatGLMClient

def get_mobilellama_client():
    from .mobilellama_client import MobileLlamaClient
    return MobileLlamaClient

def get_lora_adapter_client():
    from .lora_adapter_client import LoRAAdapterClient
    return LoRAAdapterClient

# Do not import classes directly here to avoid circular imports
__all__ = ["get_offline_model_client", "get_open_router_client", "get_os_atlas_client", "get_infigui_client", "get_ui_r1_client", "get_minicpmv_client", "get_chatglm_client", "get_mobilellama_client", "get_lora_adapter_client"]

# This file is intentionally left minimal.
# The actual clients are imported directly in model_clients.py 
