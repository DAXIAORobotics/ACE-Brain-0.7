#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Progress-estimation scorer (protocol v1): MAE of the predicted progress against the reference progress.

  python benchmarks/progress/score.py --pred predictions.jsonl [--data-dir D] [--json out.json]

Rows: {"id": ..., "pred": "<raw model text>"} or {"id": ..., "process": number}.
Parse: JSON object with numeric "process" first (last such object in the reply), then a bare number in [0,1]. Round to 2
decimals, clip to [0,1]. Missing / unparseable -> invalid: the official MAE counts error 1 for them; the valid-only MAE
ignores them. Subsets: all200 and s197 (drops the 3 questions that contain a frame with source index 320).
Lower MAE is better.
"""
import argparse, json, math, re, os, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EXCL197 = {'process-b200-0064', 'process-b200-0075', 'process-b200-0141'}
NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)'

def parse(text):
    if isinstance(text, (int, float)) and not isinstance(text, bool):
        v = float(text)
        return round(min(1., max(0., v)), 2) if math.isfinite(v) else None
    if not isinstance(text, str): return None
    t = text.split('</think>')[-1].strip()
    t = re.sub(r'^```(?:json)?|```$', '', t, flags=re.I | re.M).strip()
    for m in reversed(list(re.finditer(r'\{[^{}]*\}', t))):
        try: o = json.loads(m.group(0))
        except Exception: continue
        v = o.get('process') if isinstance(o, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
            return round(min(1., max(0., float(v))), 2)
    m = re.fullmatch(NUM, t.rstrip('.').strip())          # bare number, must already be in [0,1]
    if m and 0 <= float(m.group(0)) <= 1: return round(float(m.group(0)), 2)
    return None

def score(pred_rows, labels):
    pv = {r['id']: parse(r['pred'] if 'pred' in r else r.get('process')) for r in pred_rows}
    out = {'protocol': 'v1'}
    for name, ids in (('all200', sorted(labels)), ('s197', sorted(set(labels) - EXCL197))):
        err = {i: (abs(pv[i] - labels[i]) if pv.get(i) is not None else 1.) for i in ids}
        val = [i for i in ids if pv.get(i) is not None]
        out[name] = dict(n=len(ids), n_valid=len(val), coverage=len(val) / len(ids),
                         mae_official=sum(err.values()) / len(ids),
                         mae_valid_only=(sum(err[i] for i in val) / len(val)) if val else None)
    out['pred_mean'] = sum(v for v in pv.values() if v is not None) / max(1, sum(v is not None for v in pv.values()))
    return out, pv

def load_labels(path):
    return {d['id']: d['process'] for d in (json.loads(l) for l in open(path, encoding='utf-8') if l.strip())}

if __name__ == '__main__':
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--pred', required=True)
    a.add_argument('--data-dir', default=None, help='root holding progress/labels.jsonl (default ./data)')
    a.add_argument('--json', default=None)
    a = a.parse_args()
    rows = [json.loads(l) for l in open(a.pred, encoding='utf-8') if l.strip()]
    labels = load_labels(Path(a.data_dir or REPO / 'data') / 'progress' / 'labels.jsonl')
    s, _ = score(rows, labels)
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(s, open(a.json, 'w'), indent=1)
    A, B = s['all200'], s['s197']
    f4 = lambda v: 'n/a' if v is None else f'{v:.4f}'
    print(f"Progress MAE (all200, official; lower is better): {A['mae_official']:.4f}   [valid-only {f4(A['mae_valid_only'])}; coverage {A['coverage']:.3f} = {A['n_valid']}/{A['n']}; unparsed counted as error 1: {A['n'] - A['n_valid']}]")
    print(f"  s197: official {B['mae_official']:.4f}, valid-only {f4(B['mae_valid_only'])}   pred mean {s['pred_mean']:.3f}")
