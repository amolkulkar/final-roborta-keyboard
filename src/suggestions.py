# suggestions.py — word-level alternatives for toxic spans
import re
from typing import List, Tuple

Span = Tuple[int, int, str, str]  # (start, end, term, kind)

NONALNUM = re.compile(r"[^a-z0-9]+")
def _norm(token: str) -> str:
    t = token.lower()
    t = NONALNUM.sub("", t)
    # simple leet map
    t = (t.replace("0","o").replace("1","i").replace("3","e")
           .replace("$","s").replace("@","a").replace("!","i"))
    # collapse repeats (fuuuck -> fuuck -> f*ck class)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)
    return t

# Extend freely as you discover gaps; keep neutral, non-judgmental terms.
REPLACE_MAP = {
    # insults
    "bitch": ["person", "buddy"],
    "bastard": ["person"],
    "idiot": ["person"],
    "moron": ["person"],
    "stupid": ["unclear", "not helpful"],
    "dumb": ["unclear", "not helpful"],
    "garbage": ["nonsense", "not useful"],
    # profanity
    "fuck": ["forget"],
    "fucking": ["very"],
    "asshole": ["person"],
    "shit": ["stuff"],
    "bullshit": ["nonsense"],
    # violence / threats
    "kill": ["stop", "remove"],
    "die": ["go away"],
    # slurs (map to neutral)
    "faggot": ["person"],
    "retard": ["person"],
    "slut": ["person"],
    # sexual common
    "boob": ["term"],
    "boobs": ["terms"],
    "dick": ["term"],
    "pussy": ["term"],
    # generic fallbacks
    "bitchy": ["rude"],
    "toxic": ["unhelpful"],
}

def _choices_for_span(term: str) -> List[str]:
    key = _norm(term)
    return REPLACE_MAP.get(key, [])

def generate_suggestions(text: str, spans: List[Span]) -> List[str]:
    """
    Return only word-level alternatives. No full-sentence, no '[removed]'.
    One consolidated list for the popup.
    """
    out: List[str] = []
    seen = set()
    for _, _, term, _ in spans:
        for cand in _choices_for_span(term):
            if cand and cand not in seen:
                seen.add(cand)
                out.append(cand)
    return out
