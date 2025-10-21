import argparse, json, os
from datetime import datetime
from collections import Counter

from data_loader import as_records, prepare_for_eval
from text_sanitizer import TextSanitizer


def evaluate(dataset_path: str, limit: int = None, out_dir: str = "eval_reports"):
    # load dataset
    recs = as_records(dataset_path)
    recs = prepare_for_eval(recs)

    if limit:
        recs = recs[:limit]

    san = TextSanitizer()
    results = []
    counts = Counter()
    correct = 0
    total = 0

    for r in recs:
        text = r["text"]
        label = str(r.get("label", "")).lower()
        pred = san.analyze(text)
        action = pred["action"]

        # reduce to toxic/non-toxic for metrics
        is_toxic_pred = (action != "pass")
        is_toxic_true = (label in {"1", "toxic", "yes", "true", "sex", "toxic_sexual"})

        if is_toxic_pred == is_toxic_true:
            correct += 1
        total += 1
        counts[action] += 1

        results.append({
            "text": text,
            "label": label,
            "pred_action": action,
            "prob": pred["ml_prob"],
            "spans": pred["spans"]
        })

    acc = correct / total if total else 0.0

    # precision/recall
    tp = sum(1 for r in results if r["pred_action"] != "pass" and r["label"] in {"1","toxic","yes","true","sex","toxic_sexual"})
    fp = sum(1 for r in results if r["pred_action"] != "pass" and r["label"] not in {"1","toxic","yes","true","sex","toxic_sexual"})
    fn = sum(1 for r in results if r["pred_action"] == "pass" and r["label"] in {"1","toxic","yes","true","sex","toxic_sexual"})

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0

    report = {
        "dataset": dataset_path,
        "rows": total,
        "counts": dict(counts),
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "timestamp": datetime.now().isoformat()
    }

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, datetime.now().strftime("%Y%m%d_%H%M%S_report.json"))
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Saved report → {out_file}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="dataset CSV/JSONL")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    evaluate(args.csv, args.limit)
