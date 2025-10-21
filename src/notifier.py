# src/notifier.py
from typing import Optional

def choose_suggestion(title: str, suggestions: list[str]) -> Optional[str]:
    """
    Show suggestion buttons + a visible Skip button.
    Returns the chosen suggestion, or None for Skip/cancel.
    """
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()

        win = tk.Toplevel(root)
        win.title(title)
        win.geometry("360x240")          # taller so bottom bar is visible
        win.resizable(False, False)
        win.attributes("-topmost", True)
        win.grab_set()
        win.focus_force()

        chosen = {"val": None}
        def set_choice(opt: Optional[str]):
            chosen["val"] = opt
            try: win.grab_release()
            except Exception: pass
            win.destroy()
            root.quit()

        # title
        tk.Label(win, text="Pick a replacement or Skip:", wraplength=320, justify="left")\
          .pack(side="top", anchor="w", padx=12, pady=8)

        # suggestion buttons area (does NOT expand past bottom bar)
        list_frame = tk.Frame(win)
        list_frame.pack(side="top", fill="both", expand=True, padx=12, pady=(0, 6))

        if suggestions:
            for opt in suggestions[:8]:                # cap to avoid overflow
                tk.Button(list_frame, text=opt, command=lambda o=opt: set_choice(o))\
                  .pack(fill="x", pady=3)
        else:
            tk.Label(list_frame, text="No suggestions available.")\
              .pack(anchor="w", pady=4)

        # fixed bottom bar with Skip (always visible)
        bottom = tk.Frame(win)
        bottom.pack(side="bottom", fill="x", padx=12, pady=8)
        tk.Button(bottom, text="Skip", command=lambda: set_choice(None))\
          .pack(side="right")

        # shortcuts
        win.bind("<Escape>", lambda e: set_choice(None))
        if suggestions:
            win.bind("<Return>", lambda e: set_choice(suggestions[0]))
        else:
            win.bind("<Return>", lambda e: set_choice(None))

        win.protocol("WM_DELETE_WINDOW", lambda: set_choice(None))
        root.mainloop()
        return chosen["val"]

    except Exception:
        # console fallback
        print("\n[Notifier] Pick a replacement or Skip:")
        for i, s in enumerate(suggestions, 1):
            print(f"  {i}. {s}")
        print("  0. Skip")
        try:
            sel = int(input("Select: ").strip())
            if sel == 0: return None
            if 1 <= sel <= len(suggestions): return suggestions[sel - 1]
        except Exception:
            pass
        return None
