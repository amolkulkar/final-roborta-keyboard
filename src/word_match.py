import re
from typing import List, Tuple

# Default lists used if files are missing
DEFAULT_TOXIC  = ["bitch", "ass", "fucker", "kill", "slur", "hate"]
DEFAULT_SEXUAL = ["dick", "pussy", "cum", "boobs", "rape", "cock"]

def _load_wordlist(path: str, fallback: List[str]) -> List[str]:
    """Load newline wordlist or return fallback if file not found."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            words = [w.strip().lower() for w in f if w.strip()]
            return words if words else fallback[:]
    except FileNotFoundError:
        return fallback[:]

def _compile_word_regex(words: List[str]) -> re.Pattern:
    if not words:
        # match nothing
        return re.compile(r"(?!x)x")
    escaped = [re.escape(w) for w in words]
    return re.compile(r"\b(" + "|".join(escaped) + r")\b", re.IGNORECASE)

class WordMatcher:
    """Exact keyword matcher. Merges toxic + sexual lists."""
    def __init__(self, toxic_file: str = "data/toxic.txt", sexual_file: str = "data/sexual.txt"):
        self.toxic_words  = _load_wordlist(toxic_file,  DEFAULT_TOXIC)  if toxic_file  else []
        self.sexual_words = _load_wordlist(sexual_file, DEFAULT_SEXUAL) if sexual_file else []
        vocab = list(dict.fromkeys(self.toxic_words + self.sexual_words))  # dedupe, keep order
        self.pattern = _compile_word_regex(vocab)

    def find_spans(self, text: str) -> List[Tuple[int, int, str]]:
        return [(m.start(), m.end(), m.group(0)) for m in self.pattern.finditer(text)]

# Legacy wrapper expected by text_sanitizer.py
def find_exact(text: str, toxic_file: str = "data/toxic.txt", sexual_file: str = "data/sexual.txt"):
    wm = WordMatcher(toxic_file, sexual_file)
    return [(s, e, w, "exact") for (s, e, w) in wm.find_spans(text)]

if __name__ == "__main__":
    print(find_exact("You are a bitch and an ass"))
