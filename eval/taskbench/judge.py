"""OpenAI-compatible LLM-judge client (used by RoboBench Q1/Q2 and RoboFAC task identification).

Configuration (environment variables, or a JSON file whose path is in JUDGE_CONFIG with the same keys in lower case):
  JUDGE_BASE_URL     OpenAI-compatible base URL (required)
  JUDGE_API_KEY      API key
  JUDGE_MODEL        model name, default "qwen3.8-flash" (the judge used for the numbers in the paper)
  JUDGE_CONCURRENCY  worker threads, default 8
  JUDGE_EXTRA_BODY   JSON merged into the request body, default {"enable_thinking": false} (thinking off, as in the paper)
  JUDGE_RETRIES      transport-level retries per call, default 3

Judge back-ends are hosted services whose weights can change; re-scoring with a different back-end can move the judged
scores by a small amount. Temperature is 0 as in the paper but is not guaranteed to be bit-exact deterministic.
"""
from __future__ import annotations
import json, os, time
from concurrent.futures import ThreadPoolExecutor

DEFAULT_MODEL = "qwen3.8-flash"


def _cfg():
    cfg = {}
    p = os.environ.get("JUDGE_CONFIG")
    if p:
        cfg = json.load(open(p))
    def get(k, d=None):
        return os.environ.get("JUDGE_" + k.upper(), cfg.get(k.lower(), d))
    return get


class JudgeClient:
    def __init__(self, max_tokens=16, temperature=0.0, workers=None, cache_path=None):
        from openai import OpenAI
        get = _cfg()
        base = get("base_url")
        if not base:
            raise RuntimeError("JUDGE_BASE_URL is not set (see taskbench/judge.py)")
        self.model = get("model", DEFAULT_MODEL)
        self.max_tokens, self.temperature = max_tokens, temperature
        self.workers = int(workers or get("concurrency", 8))
        self.retries = int(get("retries", 3))
        eb = get("extra_body", '{"enable_thinking": false}')
        self.extra_body = json.loads(eb) if isinstance(eb, str) else eb
        self.client = OpenAI(base_url=base, api_key=get("api_key", "none"))

    def chat(self, messages, tag=""):
        """One chat call; raises after `retries` transport failures. `tag` is only informational."""
        last = None
        for k in range(self.retries + 1):
            try:
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens,
                    extra_body=self.extra_body or None)
                return r.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                last = e; time.sleep(min(30, 2 ** k))
        raise RuntimeError(f"judge call failed ({type(last).__name__})")

    def map(self, jobs):
        """jobs: list of (messages, tag) -> list of raw replies ('' where every retry failed)."""
        def one(j):
            try:
                return self.chat(*j)
            except Exception:  # noqa: BLE001
                return ""
        with ThreadPoolExecutor(self.workers) as ex:
            return list(ex.map(one, jobs))
