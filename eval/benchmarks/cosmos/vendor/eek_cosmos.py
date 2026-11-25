# Adapted from EmbodiedEvalKit benchmark/cosmos.py (MIT License, Copyright (c) 2025 Yifu Yuan).
# See LICENSE and NOTICE in this directory. Logic of prompt construction, answer extraction and statistics is unchanged.
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional

MC_SUFFIX = "Please answer with the letter of your choice (A, B, C, or D)."


def build_prompt(question: str, choices: List[str], instruct_following: str = "") -> str:
    """Upstream COSMOSDataset.prepare_dataset: question, newline-joined choices, fixed answer-format sentence."""
    choices_text = "\n".join(choices)
    full_prompt = f"{question}\n{choices_text}\n"
    return full_prompt + MC_SUFFIX + instruct_following


def domain_of(sample_id: str) -> str:
    return sample_id.split("_")[0] if "_" in sample_id else "unknown"


def extract_answer_from_text(text: str) -> Optional[str]:
    """
    Extract answer letter from generated text (verbatim from upstream COSMOSDataset.extract_answer_from_text).

    Supports multiple formats:
    - Answer: (A)
    - **Answer: (A)
    - Direct search for (A), (B), (C), (D)
    - Single letter A-D
    """
    text = str(text).strip()

    # Strategy 1: Try "Answer:" patterns
    answer_patterns = [
        r'(?:Answer|\u7b54\u6848)[:\s]*\(([A-Za-z])\)',  # Answer: (A)
        r'\*\*Answer[:\s]*\(([A-Za-z])\)',        # **Answer: (A)
        r'Answer[:\s]*\(([A-Za-z])\)',            # Answer: (A)
    ]

    for pattern in answer_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()

    # Strategy 2: Search for all occurrences of (A), (B), (C), (D), etc.
    all_matches = list(re.finditer(r'\(([A-Za-z])\)', text, re.IGNORECASE))
    if all_matches:
        return all_matches[-1].group(1).upper()

    # Strategy 3: Direct search for letter A-Z (word boundary)
    match = re.search(r'\b([A-Za-z])\b', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    return None


def process_raw_output(question_id: str, answer: str, raw_output_text: str) -> Dict[str, Any]:
    """Upstream process_raw_output (non-thinking mode): extract a letter and compare with the gold answer."""
    pred_choice = extract_answer_from_text(raw_output_text)
    score = 1.0 if pred_choice == answer else 0.0
    return {"question_id": question_id, "domain": domain_of(question_id), "ground_truth": answer,
            "processed_answer": pred_choice, "is_correct": score}


def compute_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Upstream compute_statistics: overall accuracy and per-domain accuracy (fractions)."""
    if not results:
        return {}
    total_score, count = 0.0, 0
    domain_stats = defaultdict(list)
    for r in results:
        score = r.get("is_correct", 0.0)
        total_score += score
        count += 1
        domain_stats[r.get("domain", "unknown")].append(score)
    overall_acc = total_score / count if count > 0 else 0.0
    final_stats = {"overall_accuracy": overall_acc}
    for domain, scores in domain_stats.items():
        final_stats[f"domain_{domain}_acc"] = sum(scores) / len(scores)
    return {"overall_score": overall_acc, "category_results": final_stats}
