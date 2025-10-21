import os, csv, json, argparse
from text_sanitizer import TextSanitizer

CANDIDATE_TEXT_COLS = ["text","comment_text","tweet","content","message","clean_text","sentence"]

def guess_text_col(headers):
    low = [h.lower() for h in headers]
    for c in CANDIDATE_TEXT_COLS:
        if c in low:
            return headers[low.index(c)]
    # fallback: first non-id-like column
    for h in headers:
        if h.lower() not in {"id","label","labels","target","toxic","toxicity"}:
            return h
    return headers[0]

def run(csv_path, text_col=None, out_path="data/dataset_predictions.jsonl", limit=None):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    san = TextSanitizer()  # uses config.yaml at project root
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    total = 0
    actions = {"pass":0,"suggest":0,"warn":0,"enforce":0}
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f, \
         open(out_path, "w", encoding="utf-8") as out:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        col = text_col or guess_text_col(headers)
        for row in reader:
            t = row.get(col, "")
            res = san.analyze(t)
            rec = {
                "input": t,
                "result": {
                    "ml_prob": float(res.get("raw_probs",{}).get("combined", res.get("ml_prob",0.0))),
                    "spans": res.get("spans", []),
                    "action": res.get("action"),
                    "suggestions": res.get("suggestions", []),
                    "sanitized": res.get("sanitized", t),
                }
            }
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            a = rec["result"]["action"]
            if a in actions: actions[a] += 1
            total += 1
            if limit and total >= limit: break
            if total % 500 == 0:
                print(f"processed {total} rows…")

    print(f"Done. rows={total}  actions={actions}  -> {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/combined_cleaned_dataset.csv")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--out", default="data/dataset_predictions.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    run(args.csv, args.text_col, args.out, args.limit)
