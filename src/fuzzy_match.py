
import re
import os
from typing import List, Tuple

WORDLIST_DIR = os.environ.get("TOXIFILTER_WORDLIST_DIR", "data/wordlists")
TOXIC_FILE = os.path.join(WORDLIST_DIR, "toxic.txt")
SEXUAL_FILE = os.path.join(WORDLIST_DIR, "sexual.txt")

LEET_MAP = {
    "a": ["a", "@", "4"],
    "b": ["b", "8"],
    "e": ["e", "3"],
    "i": ["i", "1", "l", "!"],
    "l": ["l", "1", "i"],
    "o": ["o", "0"],
    "s": ["s", r"\$","5"],
    "t": ["t", "7"],
    "g": ["g", "9"],
    "c": ["c", "("],
    "k": ["k", "|<"]
}

def read_wordlist(path: str) -> List[str]:
    if not os.path.exists(path):
        return []
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            w = line.strip()
            if not w or w.startswith("#"):
                continue
            words.append(w.lower())
    return words

def letter_group(ch: str) -> str:
    ch_low = ch.lower()
    if ch_low in LEET_MAP:
        chars = [re.escape(x) for x in LEET_MAP[ch_low]]
        if re.escape(ch_low) not in chars:
            chars.insert(0, re.escape(ch_low))
        return "(?:" + "|".join(chars) + ")"
    return re.escape(ch_low)

def make_fuzzy_regex(word: str) -> str:
    parts = []
    for ch in word:
        parts.append(letter_group(ch))
    sep = r"(?:[\W_]*)"
    pattern = sep.join(parts)
    return pattern

def compile_patterns(words: List[str]):
    pats = []
    for w in words:
        if not w or len(w) < 2:
            continue
        try:
            p = re.compile(make_fuzzy_regex(w), flags=re.IGNORECASE)
            pats.append((w, p))
        except re.error:
            pats.append((w, re.compile(re.escape(w), flags=re.IGNORECASE)))
    return pats

TOXIC_WORDS = read_wordlist(TOXIC_FILE)
SEXUAL_WORDS = read_wordlist(SEXUAL_FILE)

TOXIC_PATTERNS = compile_patterns(TOXIC_WORDS)
SEXUAL_PATTERNS = compile_patterns(SEXUAL_WORDS)

def find_fuzzy(text: str) -> List[Tuple[int, int, str, str]]:
    matches = []
    if not text:
        return matches

    for src, patterns in (("fuzzy_sexual", SEXUAL_PATTERNS), ("fuzzy_toxic", TOXIC_PATTERNS)):
        for base_word, rx in patterns:
            for m in rx.finditer(text):
                matches.append((m.start(), m.end(), m.group(0), src))
    matches.sort(key=lambda x: (x[0], -x[1]))
    return matches

if __name__ == "__main__":
    samples = [
        "You are b!i!t!c!h and a--s--s",
        "That s l u r is bad",
        "k-i-l-l threats are violent",
        "you and b i t c h.",
        "f u c k e r is written with spaces",
        "b!o!o!b!s"
    ]
    if not TOXIC_WORDS and not SEXUAL_WORDS:
        print("Warning: wordlists empty. Creating temporary example patterns.")
        TOXIC_PATTERNS = compile_patterns(["ass","bitch","fucker","slur"])
        SEXUAL_PATTERNS = compile_patterns(["sexual_term"])
    for t in samples:
        print("INPUT:", t)
        print(find_fuzzy(t))
        print("-" * 40)
