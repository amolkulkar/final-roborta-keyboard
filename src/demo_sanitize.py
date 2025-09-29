import argparse
import json
import os
from typing import List, Dict

from text_sanitizer import TextSanitizer

try:
    from data_loader import load_csv, prepare_for_eval
except Exception:
    load_csv = None
    prepare_for_eval = None

SAMPLE_SENTENCES = [
    "I hate you",
    "You are a b!i!t!c!h and an a--s--s",
    "Have a nice day",
    "That s l u r is bad",
    "I will kill you."
]


def run_on_list(san: TextSanitizer, texts: List[str]) -> List[Dict]:
    out = []
    for t in texts:
        res = san.analyze(t)
        rec = {"input": t, "result": res}
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    return out


def run_on_dataset(san: TextSanitizer, path: str, limit: int = None) -> List[Dict]:
    if load_csv is None:
        raise RuntimeError("data_loader.py not found. Put data_loader.py in repo to use --dataset mode.")
    records = load_csv(path)
    records = prepare_for_eval(records, preprocess_text=False)
    if limit:
        records = records[:limit]
    texts = [r.get("text", "") for r in records]
    out = []
    for r, t in zip(records, texts):
        res = san.analyze(t)
        rec = {
            "input": t,
            "label": r.get("label"),
            "result": res
        }
        out.append(rec)
        print(json.dumps(rec, ensure_ascii=False))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", help="Path to CSV/JSONL dataset (uses data_loader.load_csv).")
    p.add_argument("--limit", type=int, default=None, help="Limit number of samples from dataset.")
    p.add_argument("--out", default="demo_sanitize_results.json", help="Output JSON file.")
    args = p.parse_args()

    san = TextSanitizer()

    if args.dataset:
        if not os.path.exists(args.dataset):
            raise FileNotFoundError(f"Dataset not found: {args.dataset}")
        results = run_on_dataset(san, args.dataset, args.limit)
    else:
        results = run_on_list(san, SAMPLE_SENTENCES)

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {len(results)} results to {args.out}")


if __name__ == "__main__":
    main()
