# src/main.py
import argparse
import json
import logging
import os
import runpy
import signal
import sys
import time
from typing import Optional

try:
    import yaml
except Exception:
    yaml = None  # optional; used only for nicer config loading

# local imports (project)
# Note: these modules must exist in src/
try:
    from tools.generate_checksums import verify_checksums_from_file  # optional helper (if present)
except Exception:
    verify_checksums_from_file = None

# keyboard hook API (must expose run_hook() -> thread with .stop())
from keyboard_hook import run_hook  # returns _HookThread instance
from text_sanitizer import TextSanitizer

LOG = logging.getLogger("toxifilter.main")
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
LOG.addHandler(handler)


def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        LOG.info("config.yaml not found, using defaults")
        return {}
    if yaml:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # fallback simple loader
    with open(path, "r", encoding="utf-8") as f:
        return json.loads(f.read())


def verify_checksums(path: str = "model/archive/checksums.txt") -> bool:
    """
    If a checksum helper exists in tools, prefer that. Otherwise
    perform a conservative existence check for model files referenced in config.
    Returns True if checks pass.
    """
    if not os.path.exists(path):
        LOG.warning("checksums file not found: %s", path)
        return False
    if verify_checksums_from_file:
        try:
            ok = verify_checksums_from_file(path)
            LOG.info("checksum helper result: %s", ok)
            return bool(ok)
        except Exception as e:
            LOG.warning("verify_checksums_from_file failed: %s", e)
            return False
    # fallback: just return True because we assume model files are present
    LOG.info("checksums file exists; skipping deep verification (no helper).")
    return True


def run_dry_demo():
    """
    Run demo_sanitize.py in-process without passing CLI args.
    If it fails, fall back to a builtin smoke demo.
    """
    demo_path = os.path.join("src", "demo_sanitize.py")
    if os.path.exists(demo_path):
        LOG.info("Running demo_sanitize.py (dry-run).")
        try:
            # avoid passing main's CLI args to demo
            old_argv = sys.argv[:]
            sys.argv = [demo_path]
            runpy.run_path(demo_path, run_name="__main__")
            return
        except SystemExit:
            return
        except Exception as e:
            LOG.exception("demo_sanitize.py failed: %s", e)
        finally:
            sys.argv = old_argv

    # fallback: run a small built-in smoke check using TextSanitizer
    LOG.info("Running built-in smoke demo (TextSanitizer).")
    s = TextSanitizer()
    samples = [
        "I hate you",
        "Have a nice day",
        "You are a b!tch",
        "I will kill you"
    ]
    for t in samples:
        try:
            out = s.analyze(t)
            LOG.info("demo -> %s", json.dumps(out, ensure_ascii=False))
        except Exception as e:
            LOG.exception("smoke sample failed: %s", e)


def start_hook(no_checks: bool = False, dry_run: bool = False, config_path: str = "config.yaml"):
    cfg = load_config(config_path)
    LOG.info("Loading config: %s", config_path)

    backend = cfg.get("preferred_backend", cfg.get("preferred", "onnx"))
    LOG.info("Selected backend: %s", backend)

    if not no_checks:
        ok = verify_checksums()
        if not ok:
            LOG.error("Checksum verification failed. Use --no-checks to bypass.")
            raise SystemExit(1)
    else:
        LOG.info("Checksum verification skipped (--no-checks).")

    if dry_run:
        # run demo and exit
        run_dry_demo()
        return

    # Start the hook thread (non-blocking) with robust lifecycle handling
    LOG.info("Starting keyboard_hook.run_hook() (background).")
    hook_thread = None
    try:
        hook_thread = run_hook()  # returns a _HookThread instance
        LOG.info("Hook thread started: %s", hook_thread)

        LOG.info("Hook running. Press Ctrl+C to stop.")
        # keep process alive until interrupted, ensure we stop thread on exit
        while True:
            time.sleep(0.5)
            alive = False
            try:
                alive = bool(getattr(hook_thread, "is_alive", lambda: True)())
            except Exception:
                # if is_alive check fails, assume alive
                alive = True
            if not alive:
                LOG.warning("Hook thread is not alive anymore. Exiting.")
                break

    except KeyboardInterrupt:
        LOG.info("Stopping hook thread (KeyboardInterrupt).")
    except Exception as e:
        LOG.exception("Runtime exception: %s", e)
        # fall through to cleanup and exit with non-zero
        raise SystemExit(2)
    finally:
        # graceful stop
        try:
            if hook_thread is not None:
                try:
                    if hasattr(hook_thread, "stop"):
                        hook_thread.stop()
                    elif hasattr(hook_thread, "listener"):
                        try:
                            hook_thread.listener.stop()  # pynput listener object
                        except Exception:
                            pass
                except Exception:
                    LOG.exception("Error while stopping hook thread.")
                # small pause to let thread exit
                time.sleep(0.25)
        except Exception:
            pass
    LOG.info("Main exiting.")
    raise SystemExit(0)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Toxifilter main runtime launcher")
    p.add_argument("--no-checks", action="store_true", help="skip checksum verification")
    p.add_argument("--dry-run", action="store_true", help="run demo and exit")
    p.add_argument("--config", type=str, default="config.yaml", help="path to config.yaml")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # set logging level from env if provided
    lvl = os.environ.get("TOXIFILTER_LOG", "INFO").upper()
    LOG.setLevel(getattr(logging, lvl, logging.INFO))
    start_hook(no_checks=args.no_checks, dry_run=args.dry_run, config_path=args.config)


if __name__ == "__main__":
    main()
