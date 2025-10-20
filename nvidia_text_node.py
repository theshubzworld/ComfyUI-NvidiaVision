import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Generator

import torch
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NVIDIATextNode:
    """
    A ComfyUI node for NVIDIA's text generation models through their API.
    Supports both streaming and non-streaming responses for text generation.
    """
    
    def __init__(self):
        self.last_request_time: float = 0
        self.min_request_interval: float = 1.0  # Seconds between requests
        self.cache: Dict[str, str] = {}
        self._current_api_key: Optional[str] = None
        self._consecutive_errors: int = 0
        self._client: Optional[OpenAI] = None
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "model": ([
                    "meta/llama-3.3-70b-instruct",
                    "meta/llama-3.3-8b-instruct",
                    "meta/llama-3.1-8b-instruct",
                    "meta/llama-3.1-70b-instruct",
                    "meta/llama-3.1-405b-instruct",
                    "meta/llama-3.1-70b",
                    "meta/llama-3.1-405b",
                    "meta/codellama-3-70b-instruct",
                    "meta/codellama-3-8b-instruct",
                    "microsoft/phi-3.5-4k-instruct",
                    "microsoft/phi-3.5-128k-instruct",
                    "microsoft/phi-3.5-16k-instruct",
                    "mistralai/mistral-large-32k",
                    "mistralai/mistral-7b-instruct",
                    "mistralai/mixtral-8x22b-instruct",
                    "mistralai/mixtral-8x7b-instruct",
                    "google/gemma-2-27b-it",
                    "google/gemma-2-9b-it",
                    "google/gemma-1.1-7b-it",
                    "google/gemma-1.1-2b-it",
                    "nvidia/teknium-openhermes-2.5-7b",
                    "nvidia/llama3-70b-8192",
                    "nvidia/llama3-8b-8192",
                    "nvidia/llama2-70b-4096",
                    "nvidia/llama2-13b-4096",
                    "nvidia/llama2-7b-4096",
                    "nvidia/nemotron-4-340b-reward",
                    "nvidia/nemotron-4-340b-instruct",
                    "nvidia/nemotron-4-15b-instruct",
                    "nvidia/nemotron-4-3b-instruct",
                    "nvidia/llama3.1-70b",
                    "nvidia/llama3.1-8b",
                    "nvidia/llama3.1-405b",
                    "nvidia/llama3.1-70b-instruct",
                    "nvidia/llama3.1-8b-instruct",
                    "nvidia/llama3.1-405b-instruct",
                    "nvidia/llama3.1-70b-chat",
                    "nvidia/llama3.1-8b-chat",
                    "nvidia/llama3.1-405b-chat",
                ], {
                    "default": "meta/llama-3.3-70b-instruct",
                    "description": "Select the text generation model. Models with -instruct or -chat are fine-tuned for instruction following"
                }),
                "prompt": ("STRING", {
                    "default": "Write a creative and detailed prompt for an AI image generator that would create a beautiful landscape.",
                    "multiline": True,
                    "dynamicPrompts": True
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a creative prompt engineer. Generate detailed and creative prompts for AI image generation. Include specific details about composition, style, lighting, and mood.",
                    "multiline": True
                }),
                "max_tokens": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 8}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "top_p": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "frequency_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "presence_penalty": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.1}),
                "stream": ("BOOLEAN", {"default": False, "label_on": "Stream", "label_off": "Single Response"}),
                "use_cache": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Disabled"}),
                "max_retries": ("INT", {"default": 3, "min": 0, "max": 10, "step": 1}),
                "retry_delay": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 30.0, "step": 0.5}),
            },
            "optional": {
                "seed": ("INT", {"default": -1, "min": -1, "max": 2**32-1, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "generate_text"
    CATEGORY = "NVIDIA/Text"
    OUTPUT_NODE = True

    def get_client(self, api_key: str) -> OpenAI:
        """Get or create the OpenAI client with the provided API key."""
        if not api_key:
            raise ValueError("API key is required")
            
        if self._client is None or self._current_api_key != api_key:
            self._client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
            self._current_api_key = api_key
            
        return self._client

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()

    def generate_text(
        self,
        api_key: str,
        model: str,
        prompt: str,
        system_prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stream: bool,
        use_cache: bool,
        max_retries: int,
        retry_delay: float,
        seed: int = -1,
    ) -> Tuple[str]:
        """Generate text using the specified model."""
        # Create a cache key
        cache_key = f"{model}:{system_prompt}:{prompt}:{max_tokens}:{temperature}:{top_p}"
        if use_cache and cache_key in self.cache:
            logger.info("Using cached response")
            return (self.cache[cache_key],)

        # Prepare messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Prepare parameters
        params = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream,
        }
        
        if seed != -1:
            params["seed"] = seed

        # Make the request with retries
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()
                client = self.get_client(api_key)
                
                if stream:
                    # For streaming, we'll collect all chunks first
                    response = client.chat.completions.create(**params)
                    chunks = []
                    for chunk in response:
                        if chunk.choices and chunk.choices[0].delta.content is not None:
                            chunks.append(chunk.choices[0].delta.content)
                    
                    full_response = ''.join(chunks)
                    if use_cache:
                        self.cache[cache_key] = full_response
                    return (full_response,)
                else:
                    # Handle non-streaming response
                    response = client.chat.completions.create(**params)
                    content = response.choices[0].message.content
                    
                    if use_cache:
                        self.cache[cache_key] = content
                    
                    return (content,)
                    
            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                continue
        
        # If we get here, all retries failed
        error_msg = f"Failed after {max_retries + 1} attempts. Last error: {str(last_error)}"
        logger.error(error_msg)
        return (f"Error: {error_msg}",)

# Node registration
NODE_CLASS_MAPPINGS = {
    "✨ NVIDIA Prompt Generator (Text) ✨": NVIDIATextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "✨ NVIDIA Prompt Generator (Text) ✨": "✨ NVIDIA Prompt Generator (Text) ✨"
}

# For backward compatibility
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
