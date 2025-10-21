# src/fuzzy_match.py
import re
from typing import List, Tuple

Span = Tuple[int, int, str, str]  # (start, end, token, kind)

# core toxic terms (extend later via lexicon)
BASE_TOXIC = {
    "bitch","ass","fuck","idiot","bastard","slur","kill","moron","stupid","dumb"
}

LEET = str.maketrans({
    "a":"a@4","b":"b8","c":"c(","e":"e3","g":"g9","i":"i1!","l":"l1|","o":"o0",
    "s":"s$5","t":"t7","+":"t","z":"z2"
})

def _gap_regex(word: str) -> str:
    # allow up to 2 non-alnum between letters: handles b i t c h, b!tch, b---itch, etc.
    parts = []
    for ch in word:
        # leet pool for this letter
        pool = re.escape(ch) + re.escape(LEET.get(ch, ch)).replace("\\", "")
        parts.append(f"[{pool}]+")
    sep = r"[^A-Za-z0-9]{0,2}"
    return r"\b" + sep.join(parts) + r"\b"

# compile once
_PATTERNS = [re.compile(_gap_regex(w), re.IGNORECASE) for w in BASE_TOXIC]

def find_fuzzy(text: str) -> List[Span]:
    spans: List[Span] = []
    for pat in _PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end(), m.group(0), "fuzzy"))
    return spans

if __name__ == "__main__":
    print(find_fuzzy("you are a b i t c h"))
    print(find_fuzzy("that s l u r is bad"))
    print(find_fuzzy("you are a m0r0n"))
