
import re
import unicodedata

def normalize_whitespace(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

def strip_control_chars(text: str) -> str:
    return ''.join(ch for ch in text if unicodedata.category(ch)[0] != "C")

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)

def lower_and_trim(text: str) -> str:
    return normalize_whitespace(text).lower()

def collapse_repeated_chars(text: str, n: int = 3) -> str:
    return re.sub(r'(.)\1{%d,}' % n, r'\1' * n, text)

def preprocess(text: str) -> str:
    t = text
    t = strip_control_chars(t)
    t = normalize_unicode(t)
    t = collapse_repeated_chars(t, n=3)
    t = normalize_whitespace(t)
    return t

if __name__ == "__main__":
    s = "LoOOOove \n this!." \
     
    print("Input:", s)
    print("Output:", preprocess(s))
