from ReAgentV_utils.prompt_builder.prompt import (
    tool_retrieval_prompt_template,
    eval_reward_prompt_template,
    conservative_template_str,
    neutral_template_str,
    aggressive_template_str,
    meta_agent_prompt_template,
    critic_template_str
)
import json


from string import Template
from ReAgentV_utils.model_inference.model_inference import llava_inference

def evaluate_answer(question: str, answer: str, context_info: dict, video) -> list[str]:
    critic_prompt = Template(critic_template_str).substitute(
        question=question,
        answer=answer,
        context=json.dumps(context_info, indent=2)
    )

    critique_response = llava_inference(critic_prompt, video)
    if critique_response:
        return critique_response

    return []