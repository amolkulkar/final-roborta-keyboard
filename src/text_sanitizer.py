from typing import Dict, List, Tuple

from preprocess import preprocess
from word_match import find_exact
from fuzzy_match import find_fuzzy
from sentence_model import SentenceModel
from escalation import Escalator
from suggestions import generate_suggestions


Span = Tuple[int, int, str, str]  


class TextSanitizer:
    def __init__(self, config_path: str = "config.yaml",
                 th_suggest: float = 0.50, th_warn: float = 0.70, th_enforce: float = 0.90):

        self.model = SentenceModel("combined", config_path=config_path)
        self.esc = Escalator(th_suggest, th_warn, th_enforce)

    def _mask(self, raw: str, spans: List[Span], ch: str = "*") -> str:
        if not spans:
            return raw
        chars = list(raw)
        for st, ed, _, _ in spans:
            st = max(0, min(st, len(chars)))
            ed = max(st, min(ed, len(chars)))
            for i in range(st, ed):
                chars[i] = ch
        return "".join(chars)

    def analyze(self, text: str) -> Dict:
       
        pre = preprocess(text)
        spans_exact = find_exact(pre)
        spans_fuzzy = find_fuzzy(pre)
        spans: List[Span] = spans_exact + spans_fuzzy

        ml = self.model.predict(pre)
        prob = float(ml["prob"]) if isinstance(ml, dict) else float(ml[0]["prob"])

        action = self.esc.step(key=pre, prob=prob)
        level = prob  

        suggestions = generate_suggestions(text, spans)

        sanitized = self._mask(text, spans)

        return {
            "raw": text,
            "pre": pre,
            "spans": spans,
            "raw_probs": {"combined": prob},
            "ml_prob": prob,                 # ← add this
            "action": action,
            "level": level,
            "suggestions": suggestions,
            "sanitized": sanitized,
        }



if __name__ == "__main__":
    s = TextSanitizer()
    for t in [
        "I hate you",
        "You are a b!i!t!c!h and an a--s--s",
        "Have a nice day",
        "That s l u r is bad",
        "I will kill you",
    ]:
        print(s.analyze(t))
