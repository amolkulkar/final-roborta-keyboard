import re
from typing import List, Dict, Union

from preprocess import preprocess
from word_match import find_exact
from fuzzy_match import find_fuzzy
from sentence_model import SentenceModel


class TextSanitizer:
    def __init__(self, config_path: str = "config.yaml"):
        self.model = SentenceModel("combined", config_path=config_path)

    def analyze(self, text: str) -> Dict:
        pre = preprocess(text)

        spans_exact = find_exact(pre)
        spans_fuzzy = find_fuzzy(pre)
        spans = spans_exact + spans_fuzzy

        ml_out = self.model.predict(pre)
        prob = float(ml_out["prob"]) if isinstance(ml_out, dict) else float(ml_out[0]["prob"])

        if prob >= 0.9:
            action = "enforce"
        elif prob >= 0.7:
            action = "warn"
        elif prob >= 0.5:
            action = "suggest"
        else:
            action = "pass"

        sanitized = list(text)
        for (start, end, term, kind) in spans:
            for i in range(start, end):
                sanitized[i] = "*"
        sanitized = "".join(sanitized)

        return {
            "raw": text,
            "pre": pre,
            "spans": spans,
            "ml_prob": prob,
            "action": action,
            "sanitized": sanitized,
        }


if __name__ == "__main__":
    s = TextSanitizer()
    samples = [
        "You are b!tch",
        "I hate you",
        "Have a nice day",
    ]
    for t in samples:
        out = s.analyze(t)
        print("IN :", t)
        print("OUT:", out)
        print("-" * 40)
