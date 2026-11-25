# Adapted from EmbodiedEvalKit core/vllm_engine.py (MIT License, Copyright (c) 2025 Yifu Yuan).
# See LICENSE and NOTICE in this directory. Message layout, qwen_vl_utils call and sampling parameters are unchanged
# (qwen3 backbone branch only). Requires vllm, transformers, qwen_vl_utils.
from typing import Any, Dict, List

SYSTEM_PROMPT = "You are a helpful assistant."
EMPTY_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def prepare_messages(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Prepare message format for a single sample (video = list of PIL frames)."""
    content = []
    if "image" in sample and sample["image"] is not None:
        images = sample["image"] if isinstance(sample["image"], list) else [sample["image"]]
        for img in images:
            if img is not None:
                content.append({"type": "image", "image": img})
    if "video" in sample and sample["video"] is not None:
        videos = sample["video"] if isinstance(sample["video"], list) else [sample["video"]]
        for vid in videos:
            if vid is not None:
                content.append({"type": "video", "video": vid})
    content.append({"type": "text", "text": sample["question"]})
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]


def make_chat_template_fn(processor, model_type):
    """Qwen3.5 checkpoints open a <think> block by default. The kit's Qwen3-VL-Instruct models do not think, so for
    model_type 'qwen3_5' the template is rendered with enable_thinking=False and every prompt must end in the empty
    think block (same guard as the adapter used for the paper numbers). Other model types use their template as is."""
    seen = {"n": 0}

    def render(messages):
        if model_type == "qwen3_5":
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
            if not text.endswith(EMPTY_THINK):
                raise RuntimeError(f"prompt does not end in empty think block: {text[-80:]!r}")
            seen["n"] += 1
            return text
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    render.seen = seen
    return render


def prepare_vllm_input(messages, render) -> Dict[str, Any]:
    """Qwen3-VL processing of upstream VLLMInferenceEngine.prepare_vllm_input (backbone 'qwen3')."""
    from qwen_vl_utils import process_vision_info
    text = render(messages)
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs
    return {"prompt": text, "multi_modal_data": mm_data, "mm_processor_kwargs": video_kwargs}


def make_sampling_params(max_tokens=4096, temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.05,
                         presence_penalty=0.0):
    from vllm import SamplingParams
    return SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_tokens,
                          repetition_penalty=repetition_penalty, presence_penalty=presence_penalty,
                          stop_token_ids=[])
