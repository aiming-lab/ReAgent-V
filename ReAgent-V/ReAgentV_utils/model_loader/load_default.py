from transformers import (
    AutoProcessor,
    LlavaForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
    WhisperProcessor,
    WhisperForConditionalGeneration
)

import torch

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    IGNORE_INDEX
)
from llava.conversation import conv_templates, SeparatorStyle


def load_default(path_dict):
    # Required keys in path_dict
    required_keys = [
        "clip_model_path", "whisper_model_path", "llava_model_path", 
        "clip_cache_dir", "whisper_cache_dir", "llava_cache_dir"
    ]
    for key in required_keys:
        if key not in path_dict:
            raise ValueError(f"Missing required key in path_dict: {key}")

    # Load CLIP
    clip_model = CLIPModel.from_pretrained(
        path_dict["clip_model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=path_dict["clip_cache_dir"]
    )
    clip_processor = CLIPProcessor.from_pretrained(
        path_dict["clip_model_path"],
        cache_dir=path_dict["clip_cache_dir"]
    )

    # Load Whisper
    whisper_model = WhisperForConditionalGeneration.from_pretrained(
        path_dict["whisper_model_path"],
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir=path_dict["whisper_cache_dir"]
    )
    whisper_processor = WhisperProcessor.from_pretrained(
        path_dict["whisper_model_path"],
        cache_dir=path_dict["whisper_cache_dir"]
    )

    # Load LLaVA (other open-source models can also be used)
    overwrite_config = {}
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        path_dict["llava_model_path"],
        None,
        "llava_qwen",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        overwrite_config=overwrite_config
    )
    model.eval()

    # Chat template
    conv_template = "qwen_1_5"

    return {
        "clip_model": clip_model,
        "clip_processor": clip_processor,
        "whisper_model": whisper_model,
        "whisper_processor": whisper_processor,
        "tokenizer": tokenizer,
        "model": model,
        "image_processor": image_processor,
        "max_length": max_length,
        "conv_template": conv_template
    }
