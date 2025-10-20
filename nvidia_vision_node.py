import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import torch
from PIL import Image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NVIDIAResponseStream:
    def __init__(self, response):
        self.response = response
        self.buffer = b''
        
    def __iter__(self):
        for chunk in self.response.iter_lines():
            if chunk:
                yield chunk.decode('utf-8')

class NVIDIAVisionNode:
    """
    A ComfyUI node for NVIDIA's Vision models through their API.
    Supports both streaming and non-streaming responses.
    """
    
    def __init__(self):
        self.last_request_time: float = 0
        self.min_request_interval: float = 1.5  # Seconds between requests
        self.cache: Dict[str, str] = {}
        self._current_api_key: Optional[str] = None
        self._consecutive_errors: int = 0
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "model": ([
                    "meta/llama-3.2-11b-vision-instruct",
                    "meta/llama-3.2-90b-vision-instruct",
                    "meta/llama-3.2-11b-vision",
                    "meta/llama-3.2-90b-vision"
                ], {
                    "default": "meta/llama-3.2-11b-vision-instruct",
                    "description": "Select the model. -instruct versions are fine-tuned for instruction following"
                }),
                "system_prompt": ("STRING", {
                    "default": "You are an AI expert in ekphrasis, acting as a skilled art critic describing an image. Use vivid, poetic, and evocative prose in British English. Focus solely on describing the image content, style, and mood. Avoid storytelling or self-insertion. Describe all elements, including potentially uncomfortable themes if present, as art can be provocative. Every word and its order matters. The description will be used for image generation, so only include visual elements. Conclude with relevant hashtags (e.g., #ArtStyle #SubjectMatter).",
                    "multiline": True
                }),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail, focusing on its visual elements, artistic style, and overall atmosphere.",
                    "multiline": True
                }),
                "max_tokens": ("INT", {"default": 512, "min": 16, "max": 4096, "step": 8}),
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
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    FUNCTION = "process"
    CATEGORY = "NVIDIA/Vision"
    OUTPUT_NODE = True

    def encode_image(self, image_tensor: torch.Tensor) -> str:
        """Convert a PyTorch tensor to base64 encoded image."""
        try:
            # Convert tensor to numpy array and handle different formats
            if isinstance(image_tensor, torch.Tensor):
                image_array = image_tensor.cpu().numpy()
            else:
                image_array = image_tensor

            # Handle batch dimension if present
            if image_array.ndim == 4:
                image_array = image_array[0]

            # Handle channel dimension
            if image_array.shape[0] == 1:  # Grayscale
                image_array = np.squeeze(image_array, axis=0)
                pil_image = Image.fromarray((image_array * 255).astype(np.uint8), 'L').convert('RGB')
            elif image_array.shape[0] == 3:  # RGB
                pil_image = Image.fromarray((np.transpose(image_array, (1, 2, 0)) * 255).astype(np.uint8), 'RGB')
            else:  # Other formats
                pil_image = Image.fromarray((image_array * 255).astype(np.uint8))
                if pil_image.mode != 'RGB':
                    pil_image = pil_image.convert('RGB')

            # Resize if needed (optional)
            max_size = 1024
            if max(pil_image.size) > max_size:
                pil_image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

            # Convert to base64
            buffered = io.BytesIO()
            pil_image.save(buffered, format="JPEG", quality=90)
            return base64.b64encode(buffered.getvalue()).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Image encoding error: {str(e)}")
            raise ValueError(f"Failed to encode image: {str(e)}")

    def _process_stream_response(self, response) -> str:
        """Process a streaming response from the API."""
        full_response = ""
        for line in NVIDIAResponseStream(response):
            if line.startswith('data: '):
                try:
                    data = json.loads(line[6:])  # Remove 'data: ' prefix
                    if 'choices' in data and len(data['choices']) > 0:
                        delta = data['choices'][0].get('delta', {})
                        if 'content' in delta:
                            content = delta['content']
                            full_response += content
                            yield content
                except json.JSONDecodeError:
                    continue
        
        return full_response

    def process(
        self,
        api_key: str,
        model: str,
        system_prompt: str,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        stream: bool,
        use_cache: bool,
        max_retries: int,
        retry_delay: float,
        image: Optional[torch.Tensor] = None
    ) -> Tuple[str]:
        """Process the input and generate a response using NVIDIA's API."""
        # Validate API key
        if not api_key.strip():
            return ("Error: API key is required",)

        # Create cache key
        cache_key = None
        if use_cache and image is not None:
            cache_key = f"{model}:{system_prompt}:{prompt}:{hashlib.md5(image.numpy().tobytes()).hexdigest()}"
            if cache_key in self.cache:
                logger.info("Using cached response")
                return (self.cache[cache_key],)

        # Rate limiting
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        # Prepare the request
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json"
        }

        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add image if provided
        if image is not None:
            try:
                base64_image = self.encode_image(image)
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                })
            except Exception as e:
                return (f"Error processing image: {str(e)}",)
        else:
            messages.append({"role": "user", "content": prompt})

        # Prepare payload
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }

        # Make the request with retries
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                response = requests.post(
                    "https://integrate.api.nvidia.com/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    stream=stream,
                    timeout=30.0
                )
                
                response.raise_for_status()
                
                # Process the response
                if stream:
                    # For streaming, we'll collect the full response
                    full_response = ""
                    for chunk in self._process_stream_response(response):
                        full_response += chunk
                    
                    if use_cache and cache_key:
                        self.cache[cache_key] = full_response
                    
                    self.last_request_time = time.time()
                    return (full_response,)
                else:
                    # For non-streaming, just get the content
                    data = response.json()
                    content = data['choices'][0]['message']['content']
                    
                    if use_cache and cache_key:
                        self.cache[cache_key] = content
                    
                    self.last_request_time = time.time()
                    return (content,)
                    
            except requests.exceptions.RequestException as e:
                last_error = e
                logger.error(f"Request failed (attempt {attempt + 1}/{max_retries + 1}): {str(e)}")
                
                if attempt < max_retries:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time:.1f} seconds...")
                    time.sleep(wait_time)
            except Exception as e:
                last_error = e
                logger.error(f"Unexpected error: {str(e)}")
                break
        
        # If we get here, all retries failed
        error_msg = f"Failed after {max_retries + 1} attempts. Last error: {str(last_error)}"
        return (f"Error: {error_msg}",)

# Node registration
NODE_CLASS_MAPPINGS = {
    "👁️‍🗨️ NVIDIA Vision Node 👁️‍🗨️": NVIDIAVisionNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "👁️‍🗨️ NVIDIA Vision Node 👁️‍🗨️": "👁️‍🗨️ NVIDIA Vision Node 👁️‍🗨️"
}

# For backward compatibility
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
