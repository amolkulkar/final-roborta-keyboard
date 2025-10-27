# src/keyboard_hook.py
import sys
import threading
import time
from typing import List, Tuple, Dict

from text_sanitizer import TextSanitizer

Span = Tuple[int, int, str, str]  # (start, end, term, kind)

# Policy: "erase" or "mask"
MODE = "erase"
print(f"[HOOK] policy MODE={MODE}", flush=True)

# Tunables
INJECT_MUTE_MS = 200       # milliseconds to mute after injecting
ENTER_DEBOUNCE_MS = 120    # ms to ignore rapid double-enter


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


def _apply_policy(line: str, spans: List[Span]) -> str:
    return _mask_by_spans(line, spans) if MODE == "mask" else _erase_by_spans(line, spans)


def _summarize_spans(spans: List[Span]) -> Dict[str, int]:
    c = {"exact": 0, "fuzzy": 0, "ml": 0}
    for _, _, _, k in spans:
        if k in c:
            c[k] += 1
    return c


def _process_line(san: TextSanitizer, line: str):
    res = san.analyze(line)
    action = res.get("action", "pass")
    spans: List[Span] = res.get("spans", [])
    ml_prob = float(res.get("ml_prob", 0.0))

    kind_counts = _summarize_spans(spans)
    print(f"[HOOK] prob={ml_prob:.3f} action={action} spans={spans}", flush=True)
    print(
        f"[HOOK] span_sources -> exact:{kind_counts['exact']} fuzzy:{kind_counts['fuzzy']} ml:{kind_counts['ml']}",
        flush=True,
    )

    if action == "pass":
        print(f"[pass] {line}", flush=True)
        return action, line

    out = _apply_policy(line, spans) if spans else line

    # safety: downgrade enforce if no visible change
    if action == "enforce" and (not spans or out.strip() == line.strip()):
        print(f"[warn] {line}", flush=True)
        return "warn", line

    # safety: require higher prob for ML-only spans
    ml_only = bool(spans) and all(k == "ml" for *_, k in spans)
    if action == "enforce" and ml_only and ml_prob < 0.985:
        print(f"[warn] {line}", flush=True)
        return "warn", line

    print(f"[{action}] {out}", flush=True)
    return action, out


def run_cli():
    san = TextSanitizer()
    print("keyboard_hook CLI: type text, press Enter. Ctrl+C to exit.", flush=True)
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
        print("[HOOK] system listener mode (pynput ok)", flush=True)
    except Exception as e:
        import traceback

        traceback.print_exc()
        print(f"[HOOK] falling back to CLI mode (import): {type(e).__name__}: {e}", flush=True)
        t = threading.Thread(target=run_cli, daemon=True)
        t.start()
        return t

    # controller and state
    kbd = Controller()
    buffer: List[str] = []
    injecting = False
    last_hash = None
    last_enter_ts = 0.0

    # optional clipboard injector (preferred for hard targets)
    try:
        import win32clipboard as wc, win32con  # type: ignore
    except Exception:
        wc = None
        win32con = None

    def _inject_text(text: str):
        if not text:
            return
        if wc and win32con:
            try:
                wc.OpenClipboard()
                wc.EmptyClipboard()
                wc.SetClipboardData(win32con.CF_UNICODETEXT, text)
                wc.CloseClipboard()
                # paste
                kbd.press(Key.ctrl)
                kbd.press("v")
                kbd.release("v")
                kbd.release(Key.ctrl)
                return
            except Exception:
                pass
        # fallback
        kbd.type(text)

    def _enforce_in_active_window(original: str, replacement: str):
        nonlocal injecting, last_hash, buffer
        try:
            injecting = True
            # remove original characters
            for _ in original:
                kbd.press(Key.backspace)
                kbd.release(Key.backspace)
            # inject replacement reliably
            _inject_text(replacement)
            # mute briefly and flush internal buffer
            time.sleep(INJECT_MUTE_MS / 1000.0)
            buffer.clear()
            last_hash = hash(replacement or "")
        finally:
            injecting = False

    def on_press(key):
        nonlocal last_hash, injecting, last_enter_ts
        try:
            if injecting:
                return  # ignore synthetic keys

            # Enter handling with debounce
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

                # only inject when there is a change
                if action in ("warn", "enforce") and out.strip() != line.strip():
                    _enforce_in_active_window(line, out)
                return

            # backspace handling
            if key == keyboard.Key.backspace:
                if buffer:
                    buffer.pop()
                return

            # character keys
            ch = getattr(key, "char", None)
            if ch is not None:
                buffer.append(ch)

        except Exception as ex:
            with open("hook_error.log", "a", encoding="utf-8") as fh:
                fh.write(f"[on_press] {type(ex).__name__}: {ex}\n")

    def _run_and_join():
        try:
            print("[HOOK] starting listener...", flush=True)
            listener = keyboard.Listener(on_press=on_press)
            listener.start()
            print("[HOOK] listener started", flush=True)
            listener.join()
        except Exception as e:
            import traceback

            traceback.print_exc()
            with open("hook_error.log", "a", encoding="utf-8") as fh:
                fh.write(f"[listener-thread] {type(e).__name__}: {e}\n")
            print(f"[HOOK] falling back to CLI mode (listener): {type(e).__name__}: {e}", flush=True)
            run_cli()

    thread = threading.Thread(target=_run_and_join, daemon=True)

    def _stop():
        try:
            # stopping handled by listener.join exit path
            pass
        except Exception:
            pass

    # attach control attributes expected by main
    thread.listener = None  # listener is local inside thread; this is placeholder
    thread.stop = _stop
    thread.is_alive = lambda: thread.is_alive()  # safe placeholder; main should check thread existence
    thread.start()
    return thread


if __name__ == "__main__":
    run_cli()
