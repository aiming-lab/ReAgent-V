from ReAgentV import *

# Define paths for pre-trained model weights and cache directories
path_dict = {
    "clip_model_path": "/root/autodl-tmp/Video-RAG-master/LLaVA-NeXT/models/models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1",
    "clip_cache_dir": "models",
    "whisper_model_path": "/root/autodl-tmp/Video-RAG-master/LLaVA-NeXT/models/models--openai--whisper-large/snapshots/4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7",
    "whisper_cache_dir": "models",
    "llava_model_path": "/root/autodl-tmp/Video-RAG-master/LLaVA-NeXT/models/models--lmms-lab--LLaVA-Video-7B-Qwen2/snapshots/013210b3aff822f1558b166d39c1046dd109520f",
    "llava_cache_dir": "models",
}

# Load the ReAgentV system, which initializes CLIP, Whisper, LLaVA, etc.
qa_system = ReAgentV.load_default(path_dict)

# Specify the video file path and the user’s question
video_path = "CLVR_2023-08-06_Sun_Aug__6_22_58_52_2023_20655732.mp4"
question = "In the video, what is the robotic arm grabbing?"

# 1. Load and sample frames from the video:
#    - frames: all extracted frames
#    - key_frames: a subset of frames chosen for relevance
#    - key_indices: indices of those key frames
#    - max_frames_num: maximum number of frames allowed by the system
#    - raw_video: the original video object
#    - video_tensor: a tensor representation ready for the model
frames, key_frames, key_indices, max_frames_num, raw_video, video_tensor = (
    qa_system.load_and_sample_video(
        question=question,
        video_path=video_path
    )
)

# 2. Retrieve multimodal information:
#    This function decides which tools to invoke (OCR, ASR, Scene Graph, etc.)
#    based on the question and frames, and then returns:
#    - modal_info: aggregated multimodal data (e.g., detected text, audio transcripts, object detections)
#    - det_top_idx: index of the most relevant detected object in the current context
#    - USE_OCR, USE_ASR, USE_DET: booleans indicating which tools were actually used
#    “Tools can be chosen flexibly”: depending on the content of the question, different tools may be activated.
modal_info, det_top_idx, USE_OCR, USE_ASR, USE_DET = qa_system.retrieve_modal_info(
    video_path=video_path,
    question=question,
    frames=key_frames,
    raw_video=raw_video,
    clip_model=qa_system.clip_model,
    clip_processor=qa_system.clip_processor,
)

# 3. Build a multimodal prompt:
#    This function assembles a single prompt string by combining:
#    - the original question
#    - modal_info (text from OCR, transcripts from ASR, object labels from detection, etc.)
#    - det_top_idx (which object is most relevant)
#    - USE_OCR/USE_ASR/USE_DET flags: decide whether to include OCR/ASR/Detection info
#    “The prompt can be customized according to different tools”:
#    if USE_OCR=True, OCR-extracted text will be included in the prompt; if USE_DET=True,
#    object detection details will be inserted; etc.
qs = qa_system.build_multimodal_prompt(
    question=question,
    modal_info=modal_info,
    det_top_idx=det_top_idx,
    max_frames_num=max_frames_num,
    USE_DET=USE_DET,
    USE_ASR=USE_ASR,
    USE_OCR=USE_OCR,
)
print("Prompt:\n", qs)

# 4. Initial inference:
#    Send the constructed prompt (qs) and the video tensor to the LLaVA model.
#    The model will produce an initial answer based on the multimodal prompt.
initial_answer = llava_inference(qs, video_tensor)
print("Initial Answer:", initial_answer)

# 5. Generate critical (follow-up) questions:
#    Based on the initial answer and the retrieved multimodal information (modal_info),
#    this method creates follow-up or “critical” questions intended to expose
#    any blind spots or missing details in the initial response.
critique_questions = qa_system.generate_critical_questions(
    question, initial_answer, modal_info, video_tensor
)

# 6. For each critical question, dynamically select tools again and get updated modal info
updated_infos = {}
for cq in critique_questions:
    # 6.1 Create a tool-selection prompt:
    #     Insert the current critical question (cq) into a template that asks the LLaVA model
    #     which tools should be used next (e.g., “Do we need ASR? OCR? Scene Graph?”).
    tool_selection_prompt = tool_retrieval_prompt_template.format(question=cq)
    response_list = llava_inference(tool_selection_prompt, video=None)

    # 6.2 Inspect the LLaVA output to decide which tools to enable/disable:
    #     If “ASR” appears in the response, turn on USE_ASR.
    #     If “OCR” appears, turn on USE_OCR.
    #     If “Scene Graph” appears, turn on USE_DET.
    #     “Tools can be chosen flexibly”: different follow-up questions might need different tools.
    USE_ASR = "ASR" in response_list
    USE_OCR = "OCR" in response_list
    USE_DET = "Scene Graph" in response_list

    # 6.3 Retrieve multimodal info for the current critical question (cq),
    #     using whichever tools have been toggled on.
    new_modal_info, new_det_top_idx, _, _, _ = qa_system.retrieve_modal_info(
        video_path=video_path,
        question=cq,
        frames=key_frames,
        raw_video=raw_video,
        clip_model=qa_system.clip_model,
        clip_processor=qa_system.clip_processor,
    )
    # Store the updated modal info in the dictionary, keyed by the critical question.
    updated_infos[cq] = new_modal_info

# 7. Combine the original question’s modal info with each critical question’s info:
#    This context will be used to generate an evaluation report.
context_infos = {question: modal_info, **updated_infos}
context_str = json.dumps(context_infos[question], indent=2)

# 8. Generate an evaluation report:
#    Feed the original question, its modal_info, initial answer, and video tensor into
#    generate_eval_report. The report summarizes strengths, weaknesses, and missing details
#    in the initial answer.
eval_report = qa_system.generate_eval_report(
    question=question,
    context_info=context_str,
    initial_answer=initial_answer,
    video=video_tensor
)
print("Evaluation Report:", eval_report)

# 9. Produce a reflective final answer:
#    Using the original question, the initial answer, and the evaluation report,
#    get_reflective_final_answer refines or corrects the answer by “reflecting” on the feedback.
#    The model may re-examine the video tensor to produce this improved final response.
final_answer = qa_system.get_reflective_final_answer(
    question, initial_answer, eval_report, video_tensor
)
print("Final Answer:", final_answer)
