# src/test_pynput.py
from pynput import keyboard
import traceback, sys

def on_press(key):
    try:
        print("pressed:", key)
    except Exception:
        traceback.print_exc()

try:
    print("creating listener")
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    print("listener started. press keys in any window. Ctrl+C to stop.")
    listener.join()
except Exception:
    print("listener error:", file=sys.stderr)
    traceback.print_exc()
    raise
