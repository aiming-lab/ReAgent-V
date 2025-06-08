# init_modules.py

from string import Template
import requests
import os
import re
import json
import time
import copy
import torch
import yaml
import numpy as np
import torchaudio
import ffmpeg
from PIL import Image
from decord import VideoReader, cpu
import torch.nn.functional as F
import pickle
import string
import socket
import networkx as nx
import pandas as pd
from typing import Dict, List
from tqdm import tqdm


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

from ReAgentV_utils.tools.ocr_tools.rag_retriever_dynamic import retrieve_documents_with_dynamic
from ReAgentV_utils.tools.scene_graph_tools.filter_keywords import filter_keywords
from ReAgentV_utils.tools.scene_graph_tools.scene_graph import generate_scene_graph_description

from ReAgentV_utils.prompt_builder.prompt import (
    tool_retrieval_prompt_template,
    eval_reward_prompt_template,
    conservative_template_str,
    neutral_template_str,
    aggressive_template_str,
    meta_agent_prompt_template,
    critic_template_str
)

from ReAgentV_utils.frame_selection_ecrs.ECRS_frame_selection import (
    compute_entropy,
    normalize_array,
    select_keyframes
)

from ReAgentV_utils.video_processor.process_video import load_video_frames
from ReAgentV_utils.video_processor.process_video import save_frames


from ReAgentV_utils.video_processor.process_audio import (
    extract_audio,
    chunk_audio,
    transcribe_chunk
)

from ReAgentV_utils.critical_question_generator.generate_critical_question import evaluate_answer
from ReAgentV_utils.tools.audio_tools.asr_utils import get_asr_docs
from ReAgentV_utils.tools.ocr_tools.ocr_utils import get_ocr_docs

from ReAgentV_utils.tools.scene_graph_tools.det_utils import (
    calculate_xmax_ymax,
    calculate_spatial_relations,
    relation_to_text,
    generate_scene_graph_description,
    get_det_docs,
    det_preprocess
)

from ReAgentV_utils.tools.extract_modal_info import retrieve_modal_info

from ReAgentV_utils.model_loader.load_default import load_default
import ReAgentV_utils.model_inference.model_inference
from ReAgentV_utils.model_inference.model_inference import llava_inference
from ReAgentV_utils.model_loader.load_config_vars import load_config_vars
from ReAgentV_utils.prompt_builder.build_multimodal_prompt import build_multimodal_prompt
