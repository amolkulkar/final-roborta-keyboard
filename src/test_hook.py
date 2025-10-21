# src/test_hook.py
import sys, time
sys.path.insert(0, "src")
from keyboard_hook import run_hook

t = run_hook()               # start or get background hook thread
t.enqueue("you are a scumbag")
t.enqueue("i want to have sex with u")
time.sleep(1.5)
t.stop()
print("test_hook done")
