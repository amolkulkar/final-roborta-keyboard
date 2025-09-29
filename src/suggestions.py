# S2: Suggestions for replacements (regex-span aware + neutral rewrite)
from typing import List, Tuple

# If you later add wordlists, extend this map.
REPLACE_MAP = {
    "bitch": ["person", "jerk"],
    "ass": ["person", "buddy"],
    "fucker": ["person"],
    "slur": ["term"],
}

def mask_range(raw: str, start: int, end: int, ch: str="*") -> str:
    return raw[:start] + (ch * max(0, end-start)) + raw[end:]

def generate_suggestions(raw: str, spans: List[Tuple[int,int,str,str]]) -> List[str]:
    # 1) per-span soft replacements if the clean lemma is known
    out = [raw]
    txt = raw
    for (st, ed, term, kind) in spans:
        lemma = term.lower().replace(" ", "")
        repls = REPLACE_MAP.get(lemma)
        if repls:
            # first replacement option
            txt = txt[:st] + repls[0] + txt[ed:]
        else:
            # fallback mask
            txt = mask_range(txt, st, ed)
    if txt != raw:
        out.append(txt)

    # 2) global fallback options
    if not spans:
        # neutral rewrite keeps sentence but signals edit
        out.append(raw)  # original
    out.append("[removed]")  # hard removal

    # dedupe, keep order
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            uniq.append(s); seen.add(s)
    return uniq[:3]
