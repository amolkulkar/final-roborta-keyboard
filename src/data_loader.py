import csv, json, os
from typing import List, Dict, Iterable, Optional

CANDIDATE_TEXT_COLS = ["text","comment_text","tweet","content","message","clean_text","sentence","comment","body"]

def _open(path: str):
    return open(path, "r", encoding="utf-8-sig", newline="")

def load_csv(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with _open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            out.append({k.strip(): (v if v is not None else "") for k,v in row.items()})
    return out

def load_jsonl(path: str) -> List[Dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    out = []
    with _open(path) as f:
        for line in f:
            line = line.strip()
            if not line: continue
            out.append(json.loads(line))
    return out

def guess_text_col(headers: Iterable[str]) -> str:
    low = [h.lower() for h in headers]
    for c in CANDIDATE_TEXT_COLS:
        if c in low:
            return list(headers)[low.index(c)]
    for h in headers:
        hl = h.lower()
        if hl not in {"id","label","labels","target","toxicity","toxic"}:
            return h
    return list(headers)[0]

def as_records(path: str) -> List[Dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv",".tsv"]:
        return load_csv(path)
    if ext in [".jsonl",".ndjson"]:
        return load_jsonl(path)
    return load_csv(path)

def prepare_for_eval(records: List[Dict], text_col: Optional[str]=None, preprocess_text: bool=False) -> List[Dict]:
    if not records:
        return []
    headers = records[0].keys()
    col = text_col or guess_text_col(headers)
    if preprocess_text:
        from preprocess import preprocess
        for r in records:
            r["text"] = preprocess(str(r.get(col,"")))
    else:
        for r in records:
            r["text"] = str(r.get(col,""))
    for r in records:
        lbl = r.get("label", r.get("labels", r.get("target", r.get("toxicity"))))
        r["label"] = lbl
    return records

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="CSV or JSONL")
    ap.add_argument("--text-col", default=None)
    ap.add_argument("--preprocess", action="store_true")
    args = ap.parse_args()

    recs = as_records(args.path)
    recs = prepare_for_eval(recs, args.text_col, preprocess_text=args.preprocess)
    print(f"loaded: {len(recs)} rows; sample: {recs[0] if recs else '[]'}")
