#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""ERIQ scoring: rule-based multiple-choice letter (or Yes/No) match, micro accuracy over the 3,905 questions.

  python benchmarks/eriq/score.py --pred predictions.jsonl [more shard files or a run directory] [--data-dir D] [--json out.json]

Same logic as the scorer used for the paper numbers. Missing, failed or unparseable replies count as wrong; they are
reported. Overall = micro accuracy pooled over questions; macro over subsets and per-dimension micro are also printed.
"""
import argparse, collections, json, re, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def parse(resp, nopt):
    """final-answer extraction: (1) response starts with the letter / Yes|No; (2) last non-empty line is a lone letter / Yes|No (models that reason first, e.g. the base model);
    (3) last 'answer|option|choice is X' statement; (4) a single distinct standalone letter anywhere. Otherwise unparsed (= wrong)."""
    s = (resp or '').strip()
    if not s: return None
    if nopt == 0:                                    # Yes/No subset (gold 'Yes'/'No')
        t = re.sub(r'[^A-Za-z ]', ' ', s).split()
        if t and t[0].lower() in ('yes', 'no'): return t[0].capitalize()
        ln = [l for l in s.splitlines() if l.strip()][-1]; t = re.sub(r'[^A-Za-z ]', ' ', ln).split()
        if len(t) == 1 and t[0].lower() in ('yes', 'no'): return t[0].capitalize()
        m = re.findall(r'(?:answer|option|choice)(?: is|:)?\s*\W{0,3}(yes|no)\b', s, re.I)
        if m: return m[-1].capitalize()
        w = [x.lower() for x in re.sub(r'[^A-Za-z ]', ' ', s).split() if x.lower() in ('yes', 'no')]
        return w[0].capitalize() if w and len(set(w)) == 1 else None
    L = 'ABCDEF'[:max(nopt, 2)]
    m = re.match(rf'^\W{{0,3}}([{L}])(?![A-Za-z])', s)           # "A", "(A)", "A.", "**B**"
    if m: return m.group(1)
    t = re.sub(r'[^A-Za-z ]', ' ', s).split()
    if t and t[0].lower() in ('yes', 'no') and nopt == 2: return 'A' if t[0].lower() == 'yes' else 'B'   # success-detection options are A.Yes B.No
    ln = [l for l in s.splitlines() if l.strip()][-1]
    m = re.fullmatch(rf'\W{{0,4}}(?:answer\W*|option\W*)?\(?([{L}])\)?\W{{0,4}}', ln.strip(), re.I) or re.fullmatch(rf'\W{{0,4}}([{L}])[.)]\s.{{0,200}}', ln.strip())
    if m: return m.group(1).upper()
    ms = re.findall(rf'(?:answer|option|choice)(?: is|:)?\s*\W{{0,3}}([{L}])(?![A-Za-z])', s, re.I)
    if ms: return ms[-1].upper()
    ls = re.findall(rf'(?<![A-Za-z])([{L}])(?![A-Za-z])', s)
    return ls[0] if len(set(ls)) == 1 else None


def acc(rs):
    return dict(n=len(rs), acc=round(100 * sum(r['ok'] for r in rs) / len(rs), 2) if rs else None,
                n_failed=sum(r['failed'] for r in rs), n_unparsed=sum(r['pred'] is None for r in rs))


def score(pred, items):
    rows = []
    for it in items:
        p = pred.get(it['id']); failed = p is None or p.get('api_failed') or not (p.get('pred') or '').strip()
        pa = None if failed else parse(p['pred'], it['n_options'])
        rows.append(dict(id=it['id'], subset=it['subset'], dim=it['dim'], gold=it['answer'], pred=pa, ok=pa == it['answer'],
                         failed=bool(failed), n_options=it['n_options']))
    subs = sorted({r['subset'] for r in rows})
    S = dict(overall_micro=acc(rows), by_subset={s: acc([r for r in rows if r['subset'] == s]) for s in subs},
             by_dim={d: acc([r for r in rows if r['dim'] == d]) for d in sorted({r['dim'] for r in rows})})
    S['overall_macro_over_subsets'] = round(sum(v['acc'] for v in S['by_subset'].values()) / len(subs), 2)
    S['all_A_baseline_micro'] = round(100 * sum(r['gold'] == 'A' for r in rows) / len(rows), 2)
    S['random_baseline_micro'] = round(100 * sum(1 / (r['n_options'] or 2) for r in rows) / len(rows), 2)
    S['pred_letter_dist'] = dict(collections.Counter(r['pred'] for r in rows))
    return S, rows


def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(p.glob('s*/predictions.jsonl')) or sorted(p.glob('predictions*.jsonl'))
        else:
            files.append(p)
    pred = {}
    for f in files:
        for l in open(f, encoding='utf-8'):
            if l.strip():
                d = json.loads(l); pred[str(d['id'])] = d
    return pred, files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pred', nargs='+', required=True)
    ap.add_argument('--data-dir', default=None, help='root holding eriq/items.jsonl (default ./data)')
    ap.add_argument('--json', default=None)
    ap.add_argument('--allow-partial', action='store_true', help='score only the questions that have a prediction (smoke runs)')
    a = ap.parse_args()
    from taskbench.io import read_jsonl
    items = read_jsonl(Path(a.data_dir or REPO / 'data') / 'eriq' / 'items.jsonl')
    pred, files = collect(a.pred)
    if a.allow_partial:
        items = [i for i in items if i['id'] in pred]
    S, rows = score(pred, items)
    o = S['overall_micro']
    print(f"ERIQ micro accuracy: {o['acc']:.2f}  (n={o['n']}, failed/missing {o['n_failed']}, unparsed {o['n_unparsed']}; {len(files)} prediction file(s))")
    print(f"  macro over {len(S['by_subset'])} subsets: {S['overall_macro_over_subsets']}   random baseline {S['random_baseline_micro']}  all-A {S['all_A_baseline_micro']}")
    for d, v in S['by_dim'].items(): print(f"  {d:22s} {v['acc']:6.2f} (n={v['n']})")
    for s, v in S['by_subset'].items(): print(f"    {s:22s} {v['acc']:6.2f} (n={v['n']})")
    if o['n'] != 3905 and not a.allow_partial: print('  WARNING: n != 3905')
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(S, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
