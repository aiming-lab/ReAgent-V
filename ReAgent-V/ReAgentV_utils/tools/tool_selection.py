def select_tools_from_question(question):
    """
    Given a question, select which tools to use (OCR, ASR, DET) based on a model's response.

    Args:
        question (str): The input video question.
        inference_fn (callable): Function to call the model, should accept (prompt, video) as input.

    Returns:
        dict: Dictionary with keys {'USE_OCR', 'USE_ASR', 'USE_DET'} and boolean values.
    """
    from ReAgentV_utils.prompt_builder.prompt import tool_retrieval_prompt_template  # ensure this is accessible
    from ReAgentV_utils.model_inference.model_inference import llava_inference
    prompt = tool_retrieval_prompt_template.format(question=question)
    response = llava_inference(prompt, video=None)

    use_ocr = "OCR" in response
    use_asr = "ASR" in response
    use_det = "Scene Graph" in response

    return {"USE_OCR": use_ocr, "USE_ASR": use_asr, "USE_DET": use_det}
