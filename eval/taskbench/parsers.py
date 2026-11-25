"""Answer parsers shared by several benchmarks (verbatim from the evaluation code used for the paper)."""
import re


def parse_choice(p: str, letters: str = "ABCD"):
    """Final answer letter of a possibly verbose reply: bare letter, explicit 'answer' phrases, bold, else the last
    standalone option letter near the end."""
    L = f"[{letters}]"
    s = p.strip()
    if re.fullmatch(rf"\(?{L}\)?\.?", s):
        return s.strip("().")
    for pat in (rf"answer\s*(?:is|:)?\s*\**\(?({L})\b", rf"\*\*\(?({L})[\.\):\*]", rf"^\s*\(?({L})[\.\)]\s", rf"option\s+({L})\b"):
        m = re.findall(pat, s, flags=re.I | re.M)
        if m:
            return m[-1].upper()
    m = re.findall(rf"(?<![A-Za-z])({L})(?![A-Za-z])", s[-200:])
    return m[-1] if m else None
