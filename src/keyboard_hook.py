# src/keyboard_hook.py
import sys
import threading
import time
from typing import List, Tuple, Dict

from text_sanitizer import TextSanitizer
from notifier import choose_suggestion

Span = Tuple[int, int, str, str]  # (start, end, term, kind)


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
            merged[-1] = (pst, max(ped, ed), pterm, pkind if pkind != "ml" else kind)
        else:
            merged.append((st, ed, term, kind))
    return merged


def _erase_by_spans(raw: str, spans: List[Span]) -> str:
    spans = _merge_spans(spans)
    if not spans:
        return raw
    out, prev = [], 0
    for s, e, *_ in spans:
        out.append(raw[prev:s])
        prev = e
    out.append(raw[prev:])
    return "".join(out)


def _replace_spans_with(raw: str, spans: List[Span], replacement: str) -> str:
    spans = _merge_spans(spans)
    if not spans:
        return raw
    out, prev = [], 0
    for s, e, *_ in spans:
        out.append(raw[prev:s])
        out.append(replacement)
        prev = e
    out.append(raw[prev:])
    return "".join(out)


def _summarize_spans(spans: List[Span]) -> Dict[str, int]:
    c = {"exact": 0, "fuzzy": 0, "ml": 0}
    for _, _, _, k in spans:
        if k in c:
            c[k] += 1
    return c


def _process_line(san: TextSanitizer, line: str):
    """
    Analyze and decide output.
    Returns (action, output_text). output_text may be empty-string.
    """
    res = san.analyze(line)
    action = res["action"]
    spans: List[Span] = res.get("spans", [])
    suggs_raw: List[str] = res.get("suggestions", [])
    ml_prob = res.get("ml_prob", 0.0)

    kind_counts = _summarize_spans(spans)
    print(f"[HOOK] prob={ml_prob:.3f} action={action} spans={spans}")
    print(f"[HOOK] span_sources -> exact:{kind_counts['exact']} fuzzy:{kind_counts['fuzzy']} ml:{kind_counts['ml']}")

    if action == "pass":
        print(f"[pass] {line}", flush=True)
        return action, line

    masked_preview = res.get("sanitized", "")
    bad = {line, masked_preview, "[removed]", ""}
    suggestions = []
    seen = set()
    for s in suggs_raw:
        if s in bad or s in seen:
            continue
        seen.add(s)
        suggestions.append(s)

    choice = choose_suggestion(title=action.upper(), suggestions=suggestions)

    if choice is None:
        # Skip → erase toxic spans if any; if none, keep original.
        out = _erase_by_spans(line, spans) if spans else line
        print(f"[{action}] {out}", flush=True)
        return action, out
    else:
        out = _replace_spans_with(line, spans, choice)
        print(f"[{action}] {out}", flush=True)
        return action, out


def run_cli():
    san = TextSanitizer()
    print("keyboard_hook CLI: type text, press Enter. Ctrl+C to exit.")
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip("\n")
        _process_line(san, line)


def run_hook():
    """
    Start a persistent background listener thread.
    Returns a thread-like object with .stop() and .is_alive().
    Enforces by deleting original keystrokes and typing sanitized text.
    """
    san = TextSanitizer()
    try:
        from pynput import keyboard
        from pynput.keyboard import Controller, Key
    except Exception:
        # Fallback to CLI if pynput not available
        t = threading.Thread(target=run_cli, daemon=True)
        t.stop_event = threading.Event()
        t.stop = lambda: t.stop_event.set()
        t.start()
        return t

    kbd = Controller()
    buffer: List[str] = []

    def _enforce_in_active_window(original: str, replacement: str):
        # delete original characters
        for _ in original:
            kbd.press(Key.backspace)
            kbd.release(Key.backspace)
        # type replacement
        if replacement:
            kbd.type(replacement)

    def on_press(key):
        try:
            if key == keyboard.Key.enter:
                line = "".join(buffer)
                buffer.clear()
                action, out = _process_line(san, line)
                if action in ("warn", "enforce"):
                    _enforce_in_active_window(line, out)
                # For "pass" do nothing to the active window.
                return
            if key == keyboard.Key.backspace:
                if buffer:
                    buffer.pop()
                return
            ch = getattr(key, "char", None)
            if ch is not None:
                buffer.append(ch)
        except Exception as ex:
            try:
                with open("hook_error.log", "a", encoding="utf-8") as fh:
                    fh.write(f"[on_press] {type(ex).__name__}: {ex}\n")
            except Exception:
                pass

    try:
        listener = keyboard.Listener(on_press=on_press)

        def _run_and_join():
            try:
                listener.start()
                listener.join()
            except Exception as e:
                try:
                    with open("hook_error.log", "a", encoding="utf-8") as fh:
                        fh.write(f"[listener-thread] {type(e).__name__}: {e}\n")
                except Exception:
                    pass
                run_cli()

        thread = threading.Thread(target=_run_and_join, daemon=True)
    except Exception as e:
        with open("hook_error.log", "a", encoding="utf-8") as fh:
            fh.write(f"[listener-create] {type(e).__name__}: {e}\n")
        t = threading.Thread(target=run_cli, daemon=True)
        t.stop_event = threading.Event()
        t.stop = lambda: t.stop_event.set()
        t.start()
        return t

    def _stop():
        try:
            listener.stop()
        except Exception:
            pass

    thread.listener = listener
    thread.stop = _stop
    thread.is_alive = lambda: listener.is_alive()
    thread.start()
    return thread


if __name__ == "__main__":
    run_cli()
