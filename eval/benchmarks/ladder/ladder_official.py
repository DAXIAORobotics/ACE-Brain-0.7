# SPDX-License-Identifier: Apache-2.0
"""Load the OFFICIAL LADDER offline-VLM-benchmark logic (prompts, reply parsing, scoring) straight from a clone of the
official repository (github.com/insait-institute/ladder, MIT, pinned commit; see download.sh) by AST-extracting the named top-level definitions, so we never
need the bazel/ladder package environment and never restate their code. Nothing from the repo is imported or executed
except those definitions (plan.py is exec'd whole with two stub modules for its framework imports)."""
import ast, collections, dataclasses, json, re, sys, textwrap, types
from pathlib import Path
from typing import Any, Counter, Dict, List, Optional, Tuple

_SUB = Path("ladder/components/inference/api/client")
REPO = None  # set by load(repo_dir)


def _extract(path: Path, names, glb: dict, cls: str = None) -> dict:
    src = path.read_text()
    tree = ast.parse(src)
    body = tree.body
    if cls:
        body = next(n for n in body if isinstance(n, ast.ClassDef) and n.name == cls).body
    for n in body:
        nm = n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else (
            n.targets[0].id if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) else
            n.target.id if isinstance(n, ast.AnnAssign) else None)
        if nm in names:
            seg = textwrap.dedent(ast.get_source_segment(src, n))
            exec(compile(seg, f"{path.name}:{nm}", "exec"), glb)
    missing = [x for x in names if x not in glb]
    assert not missing, (path, missing)
    return glb


@dataclasses.dataclass
class Decomposition:
    subtasks: list
    ordered: bool = False
    revisits: bool = False


def load(repo_dir=None) -> types.SimpleNamespace:
    """repo_dir: root of the official git clone (default: $LADDER_REPO_DIR)."""
    global REPO
    import os
    repo_dir = repo_dir or os.environ.get("LADDER_REPO_DIR")
    if not repo_dir:
        raise SystemExit("official LADDER repo not given: pass --repo-dir or set LADDER_REPO_DIR (see benchmarks/ladder/download.sh)")
    REPO = Path(repo_dir) / _SUB
    if not REPO.is_dir():
        raise SystemExit(f"{REPO} not found: --repo-dir must be a clone of the official repo")
    # plan.py: stub its two framework imports (Episode, metric/Fraction); the scoring functions are untouched.
    stub_ep = types.ModuleType("ladder.components.inference.api.client.episode"); stub_ep.Episode = object
    stub_reg = types.ModuleType("ladder.components.inference.api.client.metrics.registry")
    stub_reg.Fraction = Tuple[float, float]; stub_reg.metric = lambda f: f
    saved = {k: sys.modules.get(k) for k in (stub_ep.__name__, stub_reg.__name__)}
    sys.modules[stub_ep.__name__], sys.modules[stub_reg.__name__] = stub_ep, stub_reg
    try:
        plan = types.ModuleType("ladder_plan"); plan.__file__ = str(REPO / "metrics/plan.py")
        exec(compile((REPO / "metrics/plan.py").read_text(), plan.__file__, "exec"), plan.__dict__)
    finally:
        for k, v in saved.items():
            if v is None: sys.modules.pop(k, None)
            else: sys.modules[k] = v
    base = {"LOGGER": types.SimpleNamespace(info=lambda *a, **k: None, warn=lambda *a, **k: None), "re": re, "json": json, "collections": collections, "Counter": Counter, "List": List, "Dict": Dict, "Any": Any,
            "Optional": Optional, "Tuple": Tuple, "Decomposition": Decomposition,
            "goal_of": plan.goal_of, "normalize": plan.normalize, "in_register": plan.in_register}
    dec = _extract(REPO / "hlc/sources/decomposers.py", ["_DECOMP_PROMPT", "_OBJECT_NAMING", "_json_plan"], dict(base))
    ver = _extract(REPO / "hlc/verifiers.py", ["_VERIFY_PROMPT", "_CONTAINER_SUFFIX", "_MULTIVIEW_SUFFIX", "_is_count"], dict(base))
    sts = _extract(REPO / "metrics/tools/score_subtask_states.py", ["classify_next", "distinct_queries", "_required"], dict(base))
    src = _extract(REPO / "hlc/sources/vlm_source.py", ["_extract_command", "_maybe_lowercase"], dict(base), cls="VLMSource")
    sd = _extract(REPO / "metrics/tools/score_decomposition.py", ["_DEPLOYED_SIZE", "_letterbox", "_as_published", "_no_plan"],
                  dict(base, np=__import__("numpy"), Image=__import__("PIL.Image", fromlist=["Image"]), Frame=Any))
    import yaml
    gen = yaml.safe_load((REPO / "hlc/configs/subtask_generator.yaml").read_text())
    cfg = types.SimpleNamespace(lowercase_command=gen["lowercase_command"])
    me = types.SimpleNamespace(config=cfg)
    me._maybe_lowercase = lambda c: src["_maybe_lowercase"](me, c)
    return types.SimpleNamespace(
        plan=plan, DECOMP_PROMPT=dec["_DECOMP_PROMPT"], OBJECT_NAMING=dec["_OBJECT_NAMING"], json_plan=dec["_json_plan"],
        VERIFY_PROMPT=ver["_VERIFY_PROMPT"], CONTAINER_SUFFIX=ver["_CONTAINER_SUFFIX"], MULTIVIEW_SUFFIX=ver["_MULTIVIEW_SUFFIX"],
        is_count=ver["_is_count"], classify_next=sts["classify_next"], distinct_queries=sts["distinct_queries"],
        required=sts["_required"], extract_command=lambda t: src["_extract_command"](me, t), NEXT_SYSTEM=gen["system_prompt"],
        NEXT_TEMPLATE="Task: {task}\nWhat should the robot do next?", letterbox=sd["_letterbox"], as_published=sd["_as_published"],
        no_plan=sd["_no_plan"], DEPLOYED=sd["_DEPLOYED_SIZE"])


def decomposition_plan(O, text: str, instruction: str):
    """Decomposer.decompose() reply handling (decomposers.py lines after complete()): JSON plan, else one subtask per line,
    else [task]. Returns (subtasks, ordered)."""
    d = O.json_plan(text)
    if d is None:
        subs = [re.sub(r"^[-*\d.)\s]+", "", l).strip() for l in text.strip().split("\n")]
        subs = [s for s in subs if s]
        d = Decomposition(subs or [instruction])
    return list(d.subtasks), bool(d.ordered)


def verifier_count(O, text: str):
    """Verifier._ask(key='count'): regex {.*} then json.loads(...).get('count'); usable iff non-negative int (not bool)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        v = json.loads(m.group(0)).get("count") if m else None
    except (json.JSONDecodeError, AttributeError):
        v = None
    return v if O.is_count(v) else None
