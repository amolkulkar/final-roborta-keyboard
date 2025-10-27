# src/keyboard_hook.py
import sys
import threading
import time
from typing import List, Tuple, Dict
from keyboard_hook import run_hook

from text_sanitizer import TextSanitizer

Span = Tuple[int, int, str, str]  # (start, end, term, kind)

# Choose how to handle toxic parts:
# "erase" → delete toxic words
# "mask"  → replace toxic words with '*'
MODE = "erase"
print(f"[HOOK] policy MODE={MODE}", flush=True)

INJECT_MUTE_MS = 200       # mute listener after we inject
ENTER_DEBOUNCE_MS = 120    # ignore rapid double-enters


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


def _mask_by_spans(raw: str, spans: List[Span], ch: str = "*") -> str:
    spans = _merge_spans(spans)
    if not spans:
        return raw
    chars = list(raw)
    for s, e, *_ in spans:
        s = max(0, min(s, len(chars)))
        e = max(s, min(e, len(chars)))
        for i in range(s, e):
            chars[i] = ch
    return "".join(chars)


def _summarize_spans(spans: List[Span]) -> Dict[str, int]:
    c = {"exact": 0, "fuzzy": 0, "ml": 0}
    for _, _, _, k in spans:
        if k in c:
            c[k] += 1
    return c


def _apply_policy(line: str, spans: List[Span]) -> str:
    if MODE == "mask":
        return _mask_by_spans(line, spans)
    return _erase_by_spans(line, spans)


def _process_line(san: TextSanitizer, line: str):
    res = san.analyze(line)
    action = res["action"]
    spans: List[Span] = res.get("spans", [])
    ml_prob = float(res.get("ml_prob", 0.0))

    kind_counts = _summarize_spans(spans)
    print(f"[HOOK] prob={ml_prob:.3f} action={action} spans={spans}")
    print(f"[HOOK] span_sources -> exact:{kind_counts['exact']} fuzzy:{kind_counts['fuzzy']} ml:{kind_counts['ml']}")

    if action == "pass":
        print(f"[pass] {line}", flush=True)
        return action, line

    out = _apply_policy(line, spans) if spans else line

    # safety 1: downgrade if nothing to change or spans empty
    if action == "enforce" and (not spans or out.strip() == line.strip()):
        print(f"[warn] {line}", flush=True)
        return "warn", line

    # safety 2: ML-only spans need higher bar
    ml_only = bool(spans) and all(k == "ml" for *_, k in spans)
    if action == "enforce" and ml_only and ml_prob < 0.985:
        print(f"[warn] {line}", flush=True)
        return "warn", line

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
    san = TextSanitizer()
    try:
        from pynput import keyboard
        from pynput.keyboard import Controller, Key
    except Exception:
        t = threading.Thread(target=run_cli, daemon=True)
        t.start()
        return t

    kbd = Controller()
    buffer: List[str] = []

    injecting = False          # mute while injecting
    last_hash = None           # skip duplicates
    last_enter_ts = 0.0        # debounce enter

    def _enforce_in_active_window(original: str, replacement: str):
        nonlocal injecting, last_hash, buffer
        try:
            injecting = True
            # erase original
            for _ in original:
                kbd.press(Key.backspace)
                kbd.release(Key.backspace)
            # type replacement
            if replacement:
                kbd.type(replacement)
            # mute and flush
            time.sleep(INJECT_MUTE_MS / 1000.0)
            buffer.clear()
            last_hash = hash(replacement or "")
        finally:
            injecting = False

    def on_press(key):
        nonlocal last_hash, injecting, last_enter_ts
        try:
            if injecting:
                return  # ignore our synthetic keys

            if key == keyboard.Key.enter:
                now = time.time()
                if now - last_enter_ts < (ENTER_DEBOUNCE_MS / 1000.0):
                    return
                last_enter_ts = now

                line = "".join(buffer)
                buffer.clear()
                if not line.strip():
                    return

                h = hash(line)
                if h == last_hash:
                    return
                last_hash = h

                action, out = _process_line(san, line)

                # skip no-op injections
                if action in ("warn", "enforce"):
                    if out.strip() == line.strip():
                        return
                    _enforce_in_active_window(line, out)
                return

            if key == keyboard.Key.backspace:
                if buffer:
                    buffer.pop()
                return

            ch = getattr(key, "char", None)
            if ch is not None:
                buffer.append(ch)

        except Exception as ex:
            with open("hook_error.log", "a", encoding="utf-8") as fh:
                fh.write(f"[on_press] {type(ex).__name__}: {ex}\n")

    from pynput import keyboard
    listener = keyboard.Listener(on_press=on_press)

    def _run_and_join():
        try:
            listener.start()
            listener.join()
        except Exception as e:
            with open("hook_error.log", "a", encoding="utf-8") as fh:
                fh.write(f"[listener-thread] {type(e).__name__}: {e}\n")
            run_cli()

    thread = threading.Thread(target=_run_and_join, daemon=True)

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

def main():
    t = run_hook()           # returns a Thread (hook listener running)
    try:
        t.join()             # keep process alive
    except KeyboardInterrupt:
        if hasattr(t, "stop"):
            t.stop()


if __name__ == "__main__":
    main()
