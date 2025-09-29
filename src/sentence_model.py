"""
sentence_model.py — single combined model (ONNX preferred)

config.yaml (project root) example:
models:
  combined:
    onnx: "models/exports/model.onnx"
    torchscript: "models/exports/model_traced.pt"
    hf_dir: "models/roberta_saved"
"""
import os
import yaml
import numpy as np
from typing import List, Union

# ONNX runtime
try:
    import onnxruntime as ort
except Exception:
    ort = None

# Torch optional
_torch_ok = False
try:
    import importlib
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # type: ignore
            _torch_ok = True
        except Exception:
            _torch_ok = False
except Exception:
    _torch_ok = False

# Tokenizer
try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


def _find_config(path: str = "config.yaml") -> str:
    # 1) env
    env = os.environ.get("TOXIFILTER_CONFIG")
    if env and os.path.exists(env):
        return os.path.abspath(env)
    # 2) as-given
    if os.path.exists(path):
        return os.path.abspath(path)
    # 3) alongside this file (src/)
    here = os.path.dirname(__file__)
    cand1 = os.path.join(here, "config.yaml")
    # 4) parent of src (project root)
    cand2 = os.path.join(os.path.dirname(here), "config.yaml")
    for c in (cand1, cand2):
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"config.yaml not found. Tried: {path}, {cand1}, {cand2}, and TOXIFILTER_CONFIG")


def load_config_and_base(path: str = "config.yaml"):
    cfg_path = _find_config(path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f), os.path.dirname(cfg_path)


def _abspath(base: str, p: str) -> str:
    if not p:
        return ""
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))


class SentenceModel:
    def __init__(self, model_key: str = "combined", config_path: str = "config.yaml"):
        if AutoTokenizer is None:
            raise RuntimeError("transformers not available. Install: pip install transformers tokenizers")

        self.cfg, self.cfg_base = load_config_and_base(config_path)
        if "models" not in self.cfg or model_key not in self.cfg["models"]:
            raise KeyError(f"models.{model_key} missing in config.yaml")

        self.model_key = model_key
        self.model_cfg = self.cfg["models"][model_key]
        self.backend = self.cfg.get("preferred_backend", "onnx")
        self.max_length = int(self.cfg.get("max_length", 128))
        self.device = self.cfg.get("device", "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.get("tokenizer_name", "roberta-base"))

        # resolve model paths relative to config.yaml location
        self.onnx_path = _abspath(self.cfg_base, self.model_cfg.get("onnx", ""))
        self.ts_path = _abspath(self.cfg_base, self.model_cfg.get("torchscript", ""))
        self.hf_dir = _abspath(self.cfg_base, self.model_cfg.get("hf_dir", ""))

        self.ort_sess = None
        self.torch_model = None
        self._load_model()

    def _load_model(self):
        # ONNX first
        if self.backend == "onnx" and ort and self.onnx_path and os.path.exists(self.onnx_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                self.ort_sess = ort.InferenceSession(
                    self.onnx_path,
                    sess_options=opts,
                    providers=["CPUExecutionProvider"]
                )
                return
            except Exception as e:
                self.ort_sess = None
                raise RuntimeError(f"ONNX load failed: {e} (path={self.onnx_path})")

        # TorchScript fallback
        if _torch_ok and self.ts_path and os.path.exists(self.ts_path):
            try:
                import torch as _torch  # type: ignore
                self.torch_model = _torch.jit.load(self.ts_path, map_location=self.device)
                self.torch_model.eval()
                return
            except Exception as e:
                self.torch_model = None
                raise RuntimeError(f"TorchScript load failed: {e} (path={self.ts_path})")

        # HF dir fallback
        if _torch_ok and self.hf_dir and os.path.isdir(self.hf_dir):
            try:
                from transformers import AutoModelForSequenceClassification
                import torch as _torch  # type: ignore
                self.torch_model = AutoModelForSequenceClassification.from_pretrained(self.hf_dir)
                self.torch_model.to(self.device)
                self.torch_model.eval()
                return
            except Exception as e:
                self.torch_model = None
                raise RuntimeError(f"HF dir load failed: {e} (dir={self.hf_dir})")

        # last try ONNX
        if ort and self.onnx_path and os.path.exists(self.onnx_path):
            try:
                self.ort_sess = ort.InferenceSession(self.onnx_path, providers=["CPUExecutionProvider"])
                return
            except Exception as e:
                self.ort_sess = None
                raise RuntimeError(f"ONNX load failed: {e} (path={self.onnx_path})")

        raise RuntimeError(
            "No valid backend for '{key}'. Paths tried:\n"
            f"  onnx: {self.onnx_path}\n"
            f"  ts  : {self.ts_path}\n"
            f"  hf  : {self.hf_dir}\n"
            "Check that these exist relative to your config.yaml."
        )

    def _prepare(self, texts: List[str]):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, text: Union[str, List[str]]):
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        enc = self._prepare(texts)

        # ONNX path
        if self.ort_sess:
            np_enc = {}
            for k, v in enc.items():
                arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
                if arr.dtype != np.int64:
                    arr = arr.astype(np.int64)
                np_enc[k] = arr

            inputs = {}
            for inp in self.ort_sess.get_inputs():
                n = inp.name  # "input_ids", "attention_mask"
                if n in np_enc:
                    inputs[n] = np_enc[n]
                elif n in ("input_mask",) and "attention_mask" in np_enc:
                    inputs[n] = np_enc["attention_mask"]
                else:
                    ref = np_enc.get("input_ids")
                    if ref is not None:
                        inputs[n] = np.zeros_like(ref, dtype=np.int64)

            out = self.ort_sess.run(None, inputs)
            logits = np.asarray(out[0])
            probs = self._softmax(logits)[:, 1]
            results = [{"logits": logits[i].tolist(), "prob": float(probs[i])} for i in range(len(texts))]
            return results[0] if single else results

        # Torch path
        if self.torch_model and _torch_ok:
            import torch as _torch  # type: ignore
            with _torch.no_grad():
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc.get("attention_mask", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    out = self.torch_model(input_ids, attention_mask)
                else:
                    out = self.torch_model(input_ids)
                if hasattr(out, "logits"):
                    logits = out.logits.cpu().numpy()
                else:
                    logits = out.cpu().numpy()
                probs = self._softmax(logits)[:, 1]
                results = [{"logits": logits[i].tolist(), "prob": float(probs[i])} for i in range(len(texts))]
                return results[0] if single else results

        raise RuntimeError("No inference backend available. Install onnxruntime or a compatible torch.")


class CombinedSentenceModel:
    def __init__(self, config_path: str = "config.yaml"):
        self.model = SentenceModel("combined", config_path=config_path)

    def predict(self, text: Union[str, List[str]]):
        out = self.model.predict(text)
        if isinstance(out, dict):
            return {"combined": out, "combined_score": float(out["prob"])}
        return [{"combined": o, "combined_score": float(o["prob"])} for o in out]


if __name__ == "__main__":
    try:
        m = SentenceModel("combined")
        print(m.predict(["I hate you", "Have a nice day"]))
    except Exception as e:
        print("Smoke test failed:", e)
