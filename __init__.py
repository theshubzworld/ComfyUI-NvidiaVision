from .nvidia_vision_node import NVIDIAVisionNode
from .nvidia_text_node import NVIDIATextNode

NODE_CLASS_MAPPINGS = {
    "✨ NVIDIA Prompt Generator (Text) ✨": NVIDIATextNode,  # NVIDIA Text Generation node with sparkles
    "👁️‍🗨️ NVIDIA Vision Node 👁️‍🗨️": NVIDIAVisionNode  # NVIDIA Vision node with eye in speech bubble
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "✨ NVIDIA Prompt Generator (Text) ✨": "✨ NVIDIA Prompt Generator (Text) ✨",
    "👁️‍🗨️ NVIDIA Vision Node 👁️‍🗨️": "👁️‍🗨️ NVIDIA Vision Node 👁️‍🗨️"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
