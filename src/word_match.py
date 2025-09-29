import re
from typing import List, Tuple
from preprocess import preprocess, lower_and_trim

WORD_LIST = [
    r"\bass\b",
    r"\bbitch\b",
    r"\bslur_example\b"
]

COMPILED = [re.compile(p, flags=re.IGNORECASE) for p in WORD_LIST]

def find_exact(text: str) -> List[Tuple[int,int,str,str]]:
    t = text
    matches = []
    for rx in COMPILED:
        for m in rx.finditer(t):
            matches.append((m.start(), m.end(), m.group(0), "exact"))
    return matches

if __name__ == "__main__":
    s = "You are a bitch and an ass"
    print(find_exact(s))
