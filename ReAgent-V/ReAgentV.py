import os
import json
import pickle
import socket
import ast
import copy

import torch
import numpy as np
import networkx as nx
from string import Template

from init_modules import *
from ReAgentV_utils.model_inference.model_inference import tokenizer as _tokenizer, model as _model
from ReAgentV_utils.prompt_builder.prompt import (
    tool_retrieval_prompt_template,
    eval_reward_prompt_template,
    conservative_template_str,
    neutral_template_str,
    aggressive_template_str,
    meta_agent_prompt_template
)
from ReAgentV_utils.tools.scene_graph_tools.det_utils import (
    save_frames,
    calculate_xmax_ymax,
    calculate_spatial_relations,
    relation_to_text,
    generate_scene_graph_description,
    get_det_docs as _get_det_docs,
    det_preprocess
)
from ReAgentV_utils.tools.audio_tools.asr_utils import get_asr_docs
from ReAgentV_utils.tools.ocr_tools.ocr_utils import get_ocr_docs
from ReAgentV_utils.frame_selection_ecrs.ECRS_frame_selection import select_keyframes
from ReAgentV_utils.video_processor.process_video import load_video_frames
from ReAgentV_utils.prompt_builder.build_multimodal_prompt import build_multimodal_prompt
from ReAgentV_utils.model_loader.load_default import load_default
from ReAgentV_utils.model_inference.model_inference import llava_inference
from ReAgentV_utils.critical_question_generator.generate_critical_question import evaluate_answer


class ReAgentV:
    def __init__(
        self,
        clip_model,
        clip_processor,
        whisper_model,
        whisper_processor,
        tokenizer,
        model,
        image_processor,
        conv_template,
    ):
        self.clip_model = clip_model
        self.clip_processor = clip_processor
        self.whisper_model = whisper_model
        self.whisper_processor = whisper_processor
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.conv_template = conv_template

        _tokenizer = tokenizer
        _model = model

    @classmethod
    def load_default(cls, path_dict: dict):
        (
            clip_model,
            clip_processor,
            whisper_model,
            whisper_processor,
            tokenizer,
            model,
            image_processor,
            conv_template,
        ) = (
            lambda m: (
                m["clip_model"],
                m["clip_processor"],
                m["whisper_model"],
                m["whisper_processor"],
                m["tokenizer"],
                m["model"],
                m["image_processor"],
                m["conv_template"],
            )
        )(modules := load_default(path_dict))

        model = model.bfloat16().to("cuda")

        from ReAgentV_utils.model_inference import model_inference

        model_inference.tokenizer = tokenizer
        model_inference.model = model

        return cls(
            clip_model,
            clip_processor,
            whisper_model,
            whisper_processor,
            tokenizer,
            model,
            image_processor,
            conv_template,
        )

    def load_and_sample_video(self, question: str, video_path: str, sample_rate: int = 1, force_sample: bool = False):

        frames = load_video_frames(video_path, sample_rate, force_sample)
        key_frames, key_indices = select_keyframes(frames, question, self.clip_model, self.clip_processor)
        max_frames_num = len(key_frames)
        raw_video = [f for f in frames]

        video_tensor = (
            self.image_processor.preprocess(key_frames, return_tensors="pt")["pixel_values"]
            .cuda()
            .bfloat16()
        )
        video_for_model = [video_tensor]

        return frames, key_frames, key_indices, max_frames_num, raw_video, video_for_model

    def retrieve_modal_info(
        self,
        video_path: str,
        question: str,
        frames: list,
        raw_video: list,
        clip_model,
        clip_processor,
    ):
       
        return retrieve_modal_info(
            video_path=video_path,
            text=question,
            frames=frames,
            raw_video=raw_video,
            clip_model=clip_model,
            clip_processor=clip_processor,
        )

    def build_multimodal_prompt(self, question: str, modal_info: dict, det_top_idx: list, max_frames_num: int, USE_DET: bool, USE_ASR: bool, USE_OCR: bool) -> str:
       
        return build_multimodal_prompt(
            text=question,
            modal_info=modal_info,
            det_top_idx=det_top_idx,
            max_frames_num=max_frames_num,
            USE_DET=USE_DET,
            USE_ASR=USE_ASR,
            USE_OCR=USE_OCR,
        )

    def generate_critical_questions(self, question: str, initial_answer: str, context_info: dict, video) -> list[str]:
       
        return evaluate_answer(question=question, answer=initial_answer, context_info=context_info, video=video)

    def generate_eval_report(self, question: str, initial_answer: str, context_info: dict, video) -> str:
       
        context_str = json.dumps(context_info, indent=2)
        critic_prompt = eval_reward_prompt_template.format(
            question=question,
            context=context_str,
            initial_answer=initial_answer,
        )
        eval_report = llava_inference(critic_prompt, video)
        return eval_report

    def get_reflective_final_answer(
        self,
        question: str,
        initial_answer: str,
        eval_report: str,
        video,
    ) -> tuple[str, dict]:
       
        neutral_template = Template(neutral_template_str).substitute(
            text=question, answer=initial_answer, eval_report=eval_report
        )
        neutral_res = llava_inference(neutral_template, video)
        try:
            neutral_parsed = json.loads(neutral_res)
            ans_neutral = neutral_parsed["final_answer"]
            conf_neutral = float(neutral_parsed["confidence"])
        except Exception:
            ans_neutral = None
            conf_neutral = 0.0

        aggressive_template = Template(aggressive_template_str).substitute(
            text=question, answer=initial_answer, eval_report=eval_report
        )
        aggressive_res = llava_inference(aggressive_template, video)
        try:
            aggr_parsed = json.loads(aggressive_res)
            ans_aggressive = aggr_parsed["final_answer"]
            conf_aggressive = float(aggr_parsed["confidence"])
        except Exception:
            ans_aggressive = None
            conf_aggressive = 0.0

        conservative_template = Template(conservative_template_str).substitute(
            text=question, answer=initial_answer, eval_report=eval_report
        )
        conservative_res = llava_inference(conservative_template, video)
        try:
            cons_parsed = json.loads(conservative_res)
            ans_conservative = cons_parsed["final_answer"]
            conf_conservative = float(cons_parsed["confidence"])
        except Exception:
            ans_conservative = None
            conf_conservative = 0.0

        meta_template = Template(meta_agent_prompt_template).substitute(
            answer_conservative=ans_conservative,
            conf_conservative=conf_conservative,
            answer_neutral=ans_neutral,
            conf_neutral=conf_neutral,
            answer_aggressive=ans_aggressive,
            conf_aggressive=conf_aggressive,
            text=question,
            initial_answer=initial_answer,
        )
        final_model_answer = llava_inference(meta_template, video)

        confidences = {
            "neutral": conf_neutral,
            "aggressive": conf_aggressive,
            "conservative": conf_conservative,
        }
        return final_model_answer

 