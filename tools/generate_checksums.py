# tools/generate_checksums.py
"""
Generate SHA256 checksums for model files and write model/archive/checksums.txt

Usage (from repo root):
    python tools/generate_checksums.py

Outputs:
    model/archive/checksums.txt
Each line:
    <sha256hex>  <relative/path>
"""
import os
import hashlib

ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__))) if __file__.endswith("generate_checksums.py") else os.getcwd()
# adjust if you place the script elsewhere
MODEL_DIRS = [
    "models/exports",
    "models",              # include roberta_saved etc
]

OUT_DIR = os.path.join("model", "archive")
OUT_FILE = os.path.join(OUT_DIR, "checksums.txt")
EXT_WHITELIST = {".onnx", ".pt", ".pth", ".bin", ".json", ".tar", ".zip"}  # include model artifacts

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def gather_files():
    files = []
    for d in MODEL_DIRS:
        p = os.path.join(ROOT, d)
        if not os.path.exists(p):
            continue
        for root, _, names in os.walk(p):
            for n in names:
                full = os.path.join(root, n)
                rel = os.path.relpath(full, ROOT).replace("\\", "/")
                _, ext = os.path.splitext(n)
                if ext.lower() in EXT_WHITELIST or "model" in n.lower() or "roberta_saved" in rel:
                    files.append((full, rel))
    # deduplicate and sort
    seen = set()
    out = []
    for full, rel in sorted(files, key=lambda x: x[1]):
        if rel in seen:
            continue
        seen.add(rel)
        out.append((full, rel))
    return out

def ensure_outdir():
    od = os.path.join(ROOT, OUT_DIR)
    os.makedirs(od, exist_ok=True)

def main():
    print("Scanning model dirs:", MODEL_DIRS)
    files = gather_files()
    if not files:
        print("No model files found in configured MODEL_DIRS. Edit the script MODEL_DIRS list if needed.")
        return
    ensure_outdir()
    lines = []
    for full, rel in files:
        try:
            hashv = sha256_file(full)
            lines.append(f"{hashv}  {rel}")
            print("OK", rel)
        except Exception as e:
            print("SKIP", rel, ":", e)
    out_path = os.path.join(ROOT, OUT_FILE)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("# Generated checksums\n")
        for L in lines:
            f.write(L + "\n")
    print("Wrote:", out_path)
    print("Lines:", len(lines))

if __name__ == "__main__":
    main()
