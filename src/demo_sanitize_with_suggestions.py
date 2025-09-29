import argparse
import json
import os
from typing import List, Dict, Tuple

from text_sanitizer import TextSanitizer

try:
    from data_loader import load_csv, prepare_for_eval
except Exception:
    load_csv = None
    prepare_for_eval = None

try:
    from suggestions import generate_suggestions 
except Exception:
    def generate_suggestions(raw: str, spans: List[Tuple[int,int,str,str]]) -> List[str]:
        masked = list(raw)
        for st, ed, _, _ in spans:
            st = max(0, st); ed = min(len(masked), ed)
            for i in range(st, ed):
                masked[i] = "*"
        masked = "".join(masked)
        return [masked, "[removed]"]

try:
    from escalation import Escalator  
except Exception:
    class Escalator:
        """Simple per-text escalation: suggest -> warn -> enforce."""
        def __init__(self, th_suggest=0.5, th_warn=0.7, th_enforce=0.9):
            self.t = (th_suggest, th_warn, th_enforce)
            self.counts = {}

        def step(self, key: str, prob: float) -> str:
            s, w, e = self.t
            base = "pass"
            if prob >= e: base = "enforce"
            elif prob >= w: base = "warn"
            elif prob >= s: base = "suggest"

            c = self.counts.get(key, 0)
            ladder = ["pass", "suggest", "warn", "enforce"]
            idx = max(ladder.index(base), min(c, 3))
            self.counts[key] = c + 1 if base != "pass" else 0
            return ladder[idx]

SAMPLES = [
    "I hate you.",
    "You are a b!i!t!c!h and an a--s--s",
    "Have a nice day",
    "That s l u r is bad",
    "I will kill you"
]

def run_list(san: TextSanitizer, esc: Escalator, texts: List[str]) -> List[Dict]:
    out = []
    for t in texts:
        res = san.analyze(t)
        prob = float(res.get("ml_prob", 0.0))
        spans = res.get("spans", [])
        action = esc.step(t, prob)
        suggs = generate_suggestions(res.get("raw", t), spans)

        rec = {
            "input": t,
            "result": {
                "ml_prob": prob,
                "spans": spans,
                "action": action,
                "suggestions": suggs,
                "sanitized": suggs[0] if suggs else res.get("sanitized", t)
            }
        }
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    return out

def run_dataset(san: TextSanitizer, esc: Escalator, path: str, limit: int = None) -> List[Dict]:
    if load_csv is None:
        raise RuntimeError("data_loader.py not found. Put data_loader.py in repo to use --dataset mode.")
    records = load_csv(path)
    records = prepare_for_eval(records, preprocess_text=False)
    if limit: records = records[:limit]
    out = []
    for r in records:
        t = r.get("text", "")
        res = san.analyze(t)
        prob = float(res.get("ml_prob", 0.0))
        spans = res.get("spans", [])
        action = esc.step(t, prob)
        suggs = generate_suggestions(res.get("raw", t), spans)
        rec = {
            "input": t,
            "label": r.get("label"),
            "result": {
                "ml_prob": prob,
                "spans": spans,
                "action": action,
                "suggestions": suggs,
                "sanitized": suggs[0] if suggs else res.get("sanitized", t)
            }
        }
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="path to CSV dataset (expects text,label columns)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="demo_sanitize_with_suggestions.json")
    args = ap.parse_args()

    san = TextSanitizer()
    esc = Escalator(th_suggest=0.5, th_warn=0.7, th_enforce=0.9)

    if args.dataset:
        if not os.path.exists(args.dataset):
            raise FileNotFoundError(args.dataset)
        results = run_dataset(san, esc, args.dataset, args.limit)
    else:
        results = run_list(san, esc, SAMPLES)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nWrote {len(results)} results to {args.out}")

if __name__ == "__main__":
    main()
