# src/text_sanitizer.py
from typing import Dict, List, Tuple
import re

from preprocess import preprocess
from word_match import find_exact
from fuzzy_match import find_fuzzy
from sentence_model import SentenceModel
from escalation import Escalator

# optional helpers from fuzzy_match
try:
    from fuzzy_match import BASE_TOXIC, _normalize_token, _ed1
except Exception:
    BASE_TOXIC = {
        "bitch", "ass", "fuck", "idiot", "bastard", "slur", "kill",
        "moron", "stupid", "dumb", "dickhead", "scumbag", "bullshit",
        "pussy", "boobs", "cock", "penis", "vagina", "fucker", "sex", "porn"
    }

    def _normalize_token(x: str) -> str:
        return re.sub(r"[^a-z0-9]", "", x.lower())

    def _ed1(a: str, b: str) -> bool:
        if a == b:
            return True
        if abs(len(a) - len(b)) > 1:
            return False
        if len(a) + 1 == len(b):
            for i in range(len(b)):
                if a == b[:i] + b[i+1:]:
                    return True
        if len(b) + 1 == len(a):
            for i in range(len(a)):
                if b == a[:i] + a[i+1:]:
                    return True
        if len(a) == len(b):
            return sum(c1 != c2 for c1, c2 in zip(a, b)) <= 1
        return False


Span = Tuple[int, int, str, str]  # (start, end, term, kind)

_STOPWORDS = {
    "i", "you", "u", "ur", "he", "she", "it", "we", "they",
    "me", "him", "her", "them", "us",
    "am", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "the", "to", "for", "of", "in", "on", "at", "by",
    "and", "or", "but", "if", "then", "else", "with", "about"
}

_HIGH_FREQ_HARMS = {
    "like", "love", "want", "have", "make", "get", "go", "see",
    "think", "know", "look", "come", "use", "take", "give", "watch", "talk"
}


def _normalize_word(tok: str) -> str:
    return _normalize_token(tok) if callable(_normalize_token) else re.sub(r"[^a-z0-9]", "", tok.lower())


def _token_is_suspicious(tok: str) -> bool:
    """
    Mark tokens likely to be the toxic term.
    - Excludes stopwords and very common verbs.
    - Excludes single-character tokens.
    - Accepts near-toxic matches, obfuscated tokens, and uncommon long tokens.
    """
    norm = _normalize_word(tok)
    if not norm:
        return False
    if len(norm) <= 1:
        return False
    if norm in _STOPWORDS or norm in _HIGH_FREQ_HARMS:
        return False

    for tox in BASE_TOXIC:
        if _ed1(norm, tox) or tox in norm or norm in tox:
            return True

    if re.search(r"[^a-z0-9]", tok.lower()):
        return True

    if len(norm) >= 5 and norm not in _HIGH_FREQ_HARMS:
        return True

    return False


def _char_span_from_word_range(raw: str, i: int, j: int) -> Tuple[int, int]:
    toks = list(re.finditer(r"\S+", raw))
    if not toks or i >= len(toks) or i >= j:
        return (0, 0)
    j = min(j, len(toks))
    return toks[i].start(), toks[j - 1].end()


def _merge_spans(spans: List[Span]) -> List[Span]:
    if not spans:
        return []
    spans = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Span] = []
    for st, ed, term, kind in spans:
        if not merged:
            merged.append((st, ed, term, kind))
            continue
        pst, ped, pterm, pkind = merged[-1]
        if st <= ped:
            merged[-1] = (pst, max(ped, ed), pterm, pkind)
        else:
            merged.append((st, ed, term, kind))
    return merged


def _refine_spans_to_tokens(raw: str, spans: List[Span]) -> List[Span]:
    """
    Replace ML spans with suspicious-token spans where possible.
    Keeps original span when no suspicious token found.
    """
    if not spans:
        return spans
    refined: List[Span] = []
    for st, ed, term, kind in spans:
        if not kind.startswith("ml"):
            refined.append((st, ed, term, kind))
            continue
        inner = raw[st:ed]
        toks = list(re.finditer(r"\S+", inner))
        token_spans: List[Span] = []
        for m in toks:
            s = st + m.start()
            e = st + m.end()
            tok = raw[s:e]
            if _token_is_suspicious(tok):
                token_spans.append((s, e, tok, "ml"))
        if token_spans:
            refined.extend(token_spans)
        else:
            # fallback: try substring match to base toxic list
            for m in re.finditer(r"\S+", inner):
                s = st + m.start()
                e = st + m.end()
                tok = raw[s:e]
                norm = _normalize_word(tok)
                if any(tox in norm or norm in tox or _ed1(norm, tox) for tox in BASE_TOXIC):
                    refined.append((s, e, tok, "ml-substr"))
                    break
            else:
                refined.append((st, ed, term, kind))
    return _merge_spans(refined)


class TextSanitizer:
    def __init__(self, config_path: str = "config.yaml"):
        self.model = SentenceModel("combined", config_path=config_path)
        self.esc = Escalator.from_config(config_path)

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

    def _ml_localize_spans(
        self,
        raw: str,
        base_prob: float,
        drop_threshold: float = 0.10,
        shrink_ratio: float = 0.8,
    ) -> List[Span]:
        """
        Candidate masking over 1..3 word windows.
        Require at least one suspicious token in window.
        Accept if prob drop > drop_threshold.
        Return top non-overlapping windows.
        """
        words = re.findall(r"\S+", raw)
        if not words:
            return []

        candidates = []
        for n in (1, 2, 3):
            for i in range(0, len(words) - n + 1):
                j = i + n
                toks = words[i:j]
                if not any(_token_is_suspicious(t) for t in toks):
                    continue
                st, ed = _char_span_from_word_range(raw, i, j)
                if ed <= st:
                    continue
                masked = raw[:st] + " the " + raw[ed:]
                masked_pre = preprocess(masked)
                p = float(self.model.predict(masked_pre)["prob"])
                drop = base_prob - p
                if drop > drop_threshold:
                    candidates.append((drop, st, ed))

        if not candidates:
            return []

        candidates.sort(key=lambda x: x[0], reverse=True)
        chosen = []
        used_ranges: List[Tuple[int, int]] = []
        for drop, st, ed in candidates:
            overlap = any(not (ed <= u_st or st >= u_ed) for u_st, u_ed in used_ranges)
            if overlap:
                continue
            chosen.append((st, ed))
            used_ranges.append((st, ed))
            if len(chosen) >= 3:
                break

        return [(st, ed, raw[st:ed], "ml") for st, ed in chosen]

    def _substring_fallback(self, raw: str) -> List[Span]:
        toks = list(re.finditer(r"\S+", raw))
        out: List[Span] = []
        for m in toks:
            tok = m.group(0)
            norm = _normalize_word(tok)
            if not norm:
                continue
            for tox in BASE_TOXIC:
                if tox in norm or norm in tox or _ed1(norm, tox):
                    out.append((m.start(), m.end(), tok, "ml-substring"))
                    break
        return _merge_spans(out)

    def _heuristic_token_pick(self, raw: str) -> List[Span]:
        toks = list(re.finditer(r"\S+", raw))
        if not toks:
            return []
        best = None
        best_score = -1
        for m in toks:
            tok = m.group(0)
            if _normalize_word(tok) in _STOPWORDS:
                continue
            score = 0
            norm = _normalize_word(tok)
            if len(norm) >= 4:
                score += 1
            if re.search(r"[^A-Za-z0-9]", tok):
                score += 2
            if score > best_score:
                best_score = score
                best = (m.start(), m.end(), tok)
        if best and best_score > 0:
            return [(best[0], best[1], best[2], "ml-heuristic")]
        return []

    def analyze(self, text: str) -> Dict:
        pre = preprocess(text)

        # regex spans
        spans_exact = find_exact(pre)
        spans_fuzzy = find_fuzzy(pre)
        spans: List[Span] = _merge_spans(spans_exact + spans_fuzzy)

        # ML score
        ml = self.model.predict(pre)
        prob = float(ml["prob"]) if isinstance(ml, dict) else float(ml[0]["prob"])

        # ML fallback when no regex spans and model is high confidence
        if not spans and prob >= self.esc.th_enforce:
            spans = self._ml_localize_spans(text, prob, drop_threshold=0.10, shrink_ratio=0.8)
            if not spans:
                spans = self._substring_fallback(text)
            if not spans and prob >= 0.98:
                spans = self._heuristic_token_pick(text)

        # refine ML spans to suspicious tokens
        spans = _refine_spans_to_tokens(text, spans)
        spans = _merge_spans(spans)

        action = self.esc.step(key=pre, prob=prob)
        level = prob
        sanitized = self._mask(text, spans)

        # Debug logs
        print("[DEBUG] Starting analysis:", text)
        print("[DEBUG] After preprocess:", pre)
        print("[DEBUG] RAW spans:", spans)
        print("[DEBUG] ML prob:", prob)
        print("[DEBUG] Escalation decision:", action)
        print("[DEBUG] Sanitized output:", sanitized)

        return {
            "raw": text,
            "pre": pre,
            "spans": spans,
            "raw_probs": {"combined": prob},
            "ml_prob": prob,
            "action": action,
            "level": level,
            "sanitized": sanitized,
        }


if __name__ == "__main__":
    s = TextSanitizer()
    tests = [
        "i want to have sex with u",
        "you are a bitch",
        "i like your boobs",
        "i love your sperm",
        "Have a nice day",
        "you are so sexy",
    ]
    for t in tests:
        print(s.analyze(t))
