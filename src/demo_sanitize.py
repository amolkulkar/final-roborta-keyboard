# src/demo_sanitize.py
import argparse, json, os, sys
from text_sanitizer import TextSanitizer

# optional dataset support
def _iter_dataset(path: str, limit: int | None):
    # lazy import to avoid hard dep when not used
    from data_loader import as_records, prepare_for_eval
    recs = prepare_for_eval(as_records(path))
    n = len(recs) if limit is None else min(len(recs), limit)
    for i in range(n):
        yield recs[i]["text"]

SAMPLES = [
    "I hate you.",
    "You are a b!i!t!c!h and an a--s--s",
    "Have a nice day",
    "That s l u r is bad",
    "I will kill you",
]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None, help="CSV/JSONL with text+label")
    ap.add_argument("--limit", type=int, default=None, help="max rows to process")
    ap.add_argument("--out", default="demo_sanitize_results.jsonl", help="output JSONL path")
    args = ap.parse_args()

    san = TextSanitizer()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    if args.dataset:
        texts = _iter_dataset(args.dataset, args.limit)
    else:
        texts = SAMPLES if args.limit is None else SAMPLES[: args.limit]

    wrote = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for t in texts:
            res = san.analyze(t)
            rec = {"input": t, "result": {
                "ml_prob": float(res.get("ml_prob", 0.0)),
                "spans": res.get("spans", []),
                "action": res.get("action"),
                "suggestions": res.get("suggestions", []),
                "sanitized": res.get("sanitized", t),
            }}
            print(json.dumps(rec, ensure_ascii=False))
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            wrote += 1

    print(f"\nWrote {wrote} results to {args.out}")

if __name__ == "__main__":
    main()
