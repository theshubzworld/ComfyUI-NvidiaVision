import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any, Generator

# It's good practice to use a library like 'openai' for interacting with NVIDIA's API,
# as it's compatible with the OpenAI SDK standard.
from openai import OpenAI

# Set up logging for better debugging and monitoring.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class NVIDIATextNode:
    """
    A ComfyUI node for generating text using NVIDIA's API.
    This node connects to the NVIDIA API endpoint to leverage their powerful
    text generation models. It supports both streaming for real-time output and
    non-streaming for complete responses, with built-in error handling and caching.
    """

    def __init__(self):
        """Initializes the node's state."""
        self.last_request_time: float = 0
        # Rate limit to prevent sending requests too frequently.
        self.min_request_interval: float = 1.0  # Seconds
        self.cache: Dict[str, str] = {}
        self._current_api_key: Optional[str] = None
        self._client: Optional[OpenAI] = None

    @classmethod
    def INPUT_TYPES(cls):
        """Defines the input types and options for the ComfyUI node."""
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False, "password": True}),
                "model": ([
                    # Top-Tier & Large Context Models
                    "meta/llama3-70b-instruct",
                    "mistralai/mixtral-8x22b-instruct",
                    "snowflake/arctic",

                    # Strong & Balanced Models
                    "google/gemma-7b",
                    "meta/llama3-8b-instruct",
                    "mistralai/mistral-7b-instruct-v0.2",

                    # Efficient & Fast Models
                    "google/gemma-2b",

                    # Deprecated / Potentially Incorrect (kept for reference, but should be verified)
                    # The following models from the original list could not be verified in the current
                    # NVIDIA API documentation and may be deprecated or have different identifiers.
                    # It's recommended to use the models listed above.
                    "nvidia/teknium-openhermes-2.5-7b",
                    "nvidia/nemotron-4-340b-reward",
                    "nvidia/nemotron-4-340b-instruct",
                ], {
                    "default": "meta/llama3-70b-instruct",
                    "description": "Select the text generation model. Models with '-instruct' are fine-tuned for instruction following."
                }),
                "prompt": ("STRING", {
                    "default": "A majestic lion surveying its kingdom from a high cliff at sunset, cinematic lighting.",
                    "multiline": True,
                    "dynamicPrompts": True
                }),
                "system_prompt": ("STRING", {
                    "default": "You are a master visual prompt engineer. When given keywords, generate a single, concise, and imaginative AI image prompt. Include only visual details: composition, perspective, depth, lighting, shadows, colours, texture, style, technique, mood, atmosphere, and focal points. Capture the richness and aesthetics a professional artist would consider. Do not include explanations, labels, step-by-step instructions, or extra text. Output only the descriptive prompt itself, in one paragraph, ready for AI image generation. you should write prompt under 300 words.",
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
        """
        Initializes or retrieves the OpenAI client for NVIDIA's API.
        A new client is created if the API key changes.
        """
        if not api_key:
            raise ValueError("NVIDIA API key is required.")
        
        if self._client is None or self._current_api_key != api_key:
            logger.info("Initializing NVIDIA API client.")
            self._client = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=api_key
            )
            self._current_api_key = api_key
            
        return self._client

    def _rate_limit(self):
        """Enforces a minimum time interval between API requests to avoid errors."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            sleep_duration = self.min_request_interval - time_since_last
            logger.info(f"Rate limiting: waiting for {sleep_duration:.2f} seconds.")
            time.sleep(sleep_duration)
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
        """
        Main function to generate text using the NVIDIA API.
        Handles caching, parameter construction, and retry logic.
        """
        cache_key = json.dumps({
            "model": model, "prompt": prompt, "system_prompt": system_prompt,
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p, "seed": seed
        }, sort_keys=True)
        
        if use_cache and cache_key in self.cache:
            logger.info("Returning cached response.")
            return (self.cache[cache_key],)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

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

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                self._rate_limit()
                client = self.get_client(api_key)
                
                logger.info(f"Requesting completion from model: {model}")
                response = client.chat.completions.create(**params)
                
                if stream:
                    chunks = [chunk.choices[0].delta.content for chunk in response if chunk.choices and chunk.choices[0].delta.content is not None]
                    full_response = ''.join(chunks)
                else:
                    full_response = response.choices[0].message.content

                if use_cache:
                    self.cache[cache_key] = full_response
                
                logger.info("Successfully received response from API.")
                return (full_response,)
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
        
        error_msg = f"Failed to generate text after {max_retries + 1} attempts. Last error: {last_error}"
        logger.error(error_msg)
        return (f"Error: {error_msg}",)

# ComfyUI Node Mappings
NODE_CLASS_MAPPINGS = {
    "✨ NVIDIA Prompt Generator (Text) ✨": NVIDIATextNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "✨ NVIDIA Prompt Generator (Text) ✨": "✨ NVIDIA Prompt Generator (Text) ✨"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
