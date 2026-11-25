"""Inference backends: local vLLM, local HuggingFace transformers, any OpenAI-compatible API.

All backends take a list of requests (chat messages from taskbench.io.item_messages) and return
[{"pred": str, "finish_reason": str, "n_tokens": int}, ...] in the same order. Decoding is greedy
(temperature 0) unless a benchmark config says otherwise.
"""
from __future__ import annotations
import base64, io, os, time
from dataclasses import dataclass, field


@dataclass
class GenConfig:
    max_tokens: int = 1024
    temperature: float = 0.0
    thinking: bool = False            # chat-template enable_thinking flag (Qwen3.x templates)
    repetition_penalty: float | None = None
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 0
    min_pixels: int = 65536           # visual token budget (pixels per image), Qwen-VL processors
    max_pixels: int = 16777216
    max_model_len: int = 32768
    max_images: int = 40
    batch_size: int = 64
    extra: dict = field(default_factory=dict)


def _open_rgb(path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("RGB")


class VLLMBackend:
    """Offline vLLM `LLM.chat`, one process per GPU (use --shard to data-parallelise across GPUs)."""

    def __init__(self, model, gen: GenConfig, tensor_parallel_size=1, gpu_memory_utilization=0.85):
        from vllm import LLM, SamplingParams
        self.gen, self.SamplingParams = gen, SamplingParams
        self.llm = LLM(model=model, max_model_len=gen.max_model_len, limit_mm_per_prompt={"image": gen.max_images},
                       mm_processor_kwargs={"min_pixels": gen.min_pixels, "max_pixels": gen.max_pixels},
                       gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True, seed=gen.seed,
                       tensor_parallel_size=tensor_parallel_size, enable_prefix_caching=True)

    def _sampling(self):
        g = self.gen
        kw = dict(temperature=g.temperature, max_tokens=g.max_tokens)
        if g.repetition_penalty is not None:
            kw["repetition_penalty"] = g.repetition_penalty
        if g.temperature > 0:
            kw.update(top_p=g.top_p, top_k=g.top_k)
        return self.SamplingParams(**kw)

    def generate(self, requests):
        sp, out = self._sampling(), []
        B = self.gen.batch_size
        for s in range(0, len(requests), B):
            msgs, imgs = [], []
            for m in requests[s:s + B]:
                conv = []
                for turn in m:
                    if isinstance(turn["content"], str):
                        conv.append(turn)
                        continue
                    parts = []
                    for c in turn["content"]:
                        if c["type"] == "image":
                            im = _open_rgb(c["path"]); imgs.append(im)
                            parts.append({"type": "image_pil", "image_pil": im})
                        else:
                            parts.append(c)
                    conv.append({"role": turn["role"], "content": parts})
                msgs.append(conv)
            try:
                res = self.llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": self.gen.thinking}, use_tqdm=False)
            finally:
                for im in imgs:
                    im.close()
            for o in res:
                c = o.outputs[0]
                out.append({"pred": c.text, "finish_reason": c.finish_reason, "n_tokens": len(c.token_ids)})
            print(f"INFERENCE_PROGRESS {len(out)}/{len(requests)}", flush=True)
        return out


class HFBackend:
    """transformers `generate` (greedy). Slower than vLLM; useful where vLLM does not support the model."""

    def __init__(self, model, gen: GenConfig, device_map="auto", dtype="bfloat16"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.gen, self.torch = gen, torch
        self.proc = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.proc.tokenizer.padding_side = "left"
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, torch_dtype=getattr(torch, dtype), device_map=device_map, trust_remote_code=True).eval()

    def generate(self, requests):
        g, torch, out = self.gen, self.torch, []
        B = max(1, min(g.batch_size, 8))
        for s in range(0, len(requests), B):
            texts, images = [], []
            for m in requests[s:s + B]:
                conv = []
                for turn in m:
                    if isinstance(turn["content"], str):
                        conv.append(turn); continue
                    parts = []
                    for c in turn["content"]:
                        if c["type"] == "image":
                            im = _open_rgb(c["path"]); images.append(im); parts.append({"type": "image"})
                        else:
                            parts.append(c)
                    conv.append({"role": turn["role"], "content": parts})
                texts.append(self.proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False,
                                                           enable_thinking=g.thinking))
            kw = {"images": images} if images else {}
            if images:
                kw["images_kwargs"] = {"min_pixels": g.min_pixels, "max_pixels": g.max_pixels}
            inp = self.proc(text=texts, padding=True, return_tensors="pt", **kw).to(self.model.device)
            gk = dict(max_new_tokens=g.max_tokens, do_sample=g.temperature > 0)
            if g.repetition_penalty is not None:
                gk["repetition_penalty"] = g.repetition_penalty
            with torch.no_grad():
                ids = self.model.generate(**inp, **gk)
            ids = ids[:, inp["input_ids"].shape[1]:]
            for row in ids:
                n = int((row != self.proc.tokenizer.pad_token_id).sum())
                out.append({"pred": self.proc.tokenizer.decode(row, skip_special_tokens=True),
                            "finish_reason": "length" if n >= g.max_tokens else "stop", "n_tokens": n})
            print(f"INFERENCE_PROGRESS {len(out)}/{len(requests)}", flush=True)
        return out


class OpenAIBackend:
    """Any OpenAI-compatible chat endpoint. Env: MODEL_BASE_URL / MODEL_API_KEY (or OPENAI_BASE_URL / OPENAI_API_KEY)."""

    def __init__(self, model, gen: GenConfig, workers=8, retries=3):
        from openai import OpenAI
        self.model, self.gen, self.workers, self.retries = model, gen, workers, retries
        self.client = OpenAI(base_url=os.environ.get("MODEL_BASE_URL") or os.environ.get("OPENAI_BASE_URL"),
                             api_key=os.environ.get("MODEL_API_KEY") or os.environ.get("OPENAI_API_KEY") or "none")

    @staticmethod
    def _data_url(path):
        im = _open_rgb(path); buf = io.BytesIO(); im.save(buf, format="JPEG", quality=90)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    def _one(self, m):
        conv = []
        for turn in m:
            if isinstance(turn["content"], str):
                conv.append(turn); continue
            parts = [{"type": "image_url", "image_url": {"url": self._data_url(c["path"])}} if c["type"] == "image"
                     else {"type": "text", "text": c["text"]} for c in turn["content"]]
            conv.append({"role": turn["role"], "content": parts})
        err = None
        for k in range(self.retries + 1):
            try:
                r = self.client.chat.completions.create(model=self.model, messages=conv, temperature=self.gen.temperature,
                                                        max_tokens=self.gen.max_tokens)
                ch = r.choices[0]
                return {"pred": ch.message.content or "", "finish_reason": ch.finish_reason,
                        "n_tokens": getattr(r.usage, "completion_tokens", 0)}
            except Exception as e:  # noqa: BLE001
                err = type(e).__name__; time.sleep(2 ** k)
        return {"pred": "", "finish_reason": "error", "n_tokens": 0, "error": err}

    def generate(self, requests):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(self._one, requests))


class MockBackend:
    """Returns a fixed reply; used by tests to exercise data loading, prompts and scoring without a model."""

    def __init__(self, model, gen, reply="A"):
        self.reply = reply

    def generate(self, requests):
        return [{"pred": self.reply, "finish_reason": "stop", "n_tokens": 1} for _ in requests]


def make_backend(kind, model, gen, **kw):
    return {"vllm": VLLMBackend, "hf": HFBackend, "openai": OpenAIBackend, "mock": MockBackend}[kind](model, gen, **kw)
