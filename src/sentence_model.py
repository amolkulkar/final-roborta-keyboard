import os
import yaml
import numpy as np
from typing import List, Union

try:
    import onnxruntime as ort
except Exception:
    ort = None

_torch_available = False
try:
    import importlib
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  
            _torch_available = True
        except Exception:
            _torch_available = False
except Exception:
    _torch_available = False

try:
    from transformers import AutoTokenizer
except Exception as ex:
    AutoTokenizer = None
    _transformers_error = ex

def load_config(path: str = "config.yaml") -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class SentenceModel:
    def __init__(self, model_key: str = "combined", config_path: str = "config.yaml"):
        self.cfg = load_config(config_path)
        if "models" not in self.cfg or model_key not in self.cfg["models"]:
            raise KeyError(f"Model key '{model_key}' not found in config.yaml models section.")
        self.model_key = model_key
        self.model_cfg = self.cfg["models"][model_key]
        self.backend = self.cfg.get("preferred_backend", "onnx")
        self.max_length = int(self.cfg.get("max_length", 128))
        self.device = self.cfg.get("device", "cpu")

        if AutoTokenizer is None:
            raise RuntimeError(
                "transformers.AutoTokenizer not available. Install a compatible transformers package.\n"
                "Recommended (safe) command:\n"
                "  pip install 'transformers==4.44.2' --no-deps\n"
                "Then install minimal dependencies:\n"
                "  pip install tokenizers\n"
                "If you see heavy packages (scikit-learn/scipy) being pulled, uninstall them:\n"
                "  pip uninstall -y scikit-learn scipy\n"
            )

        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.get("tokenizer_name", "roberta-base"))
        self.ort_sess = None
        self.torch_model = None
        self._load_model()

    def _load_model(self):
        onnx_path = self.model_cfg.get("onnx", "")
        ts_path = self.model_cfg.get("torchscript", "")
        hf_dir = self.model_cfg.get("hf_dir", None)

        if self.backend == "onnx" and ort and onnx_path and os.path.exists(onnx_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 1
                self.ort_sess = ort.InferenceSession(onnx_path, sess_options=opts)
                return
            except Exception as e:
                self.ort_sess = None

        if _torch_available and ts_path and os.path.exists(ts_path):
            try:
                import torch as _torch  
                self.torch_model = _torch.jit.load(ts_path, map_location=self.device)
                self.torch_model.eval()
                return
            except Exception:
                self.torch_model = None

        if hf_dir and os.path.isdir(hf_dir) and _torch_available:
            try:
                from transformers import AutoModelForSequenceClassification
                import torch as _torch  
                self.torch_model = AutoModelForSequenceClassification.from_pretrained(hf_dir)
                self.torch_model.to(self.device)
                self.torch_model.eval()
                return
            except Exception:
                self.torch_model = None

        if ort and onnx_path and os.path.exists(onnx_path) and not self.ort_sess:
            try:
                self.ort_sess = ort.InferenceSession(onnx_path)
                return
            except Exception:
                self.ort_sess = None

        raise RuntimeError(
            f"No valid model backend available for key '{self.model_key}'. "
            f"Checked ONNX ({bool(ort)}), Torch ({_torch_available}). "
            f"Paths tried: onnx='{onnx_path}', torchscript='{ts_path}', hf_dir='{hf_dir}'."
        )

    def _prepare(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return enc

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    def predict(self, text: Union[str, List[str]]):
        single = False
        if isinstance(text, str):
            texts = [text]
            single = True
        else:
            texts = list(text)

        enc = self._prepare(texts)

        if self.ort_sess:
            inputs = {}
            for inp in self.ort_sess.get_inputs():
                name = inp.name
                if name in enc:
                    tensor = enc[name]
                    if hasattr(tensor, "detach"):
                        arr = tensor.detach().cpu().numpy()
                    else:
                        arr = np.asarray(tensor)
                    inputs[name] = arr
            out = self.ort_sess.run(None, inputs)
            logits = np.asarray(out[0])
            probs = self._softmax(logits)[:, 1]
            results = [{"logits": logits[i].tolist(), "prob": float(probs[i])} for i in range(len(texts))]
            return results[0] if single else results

        if self.torch_model and _torch_available:
            import torch as _torch  
            with _torch.no_grad():
                input_ids = enc["input_ids"].to(self.device)
                attention_mask = enc.get("attention_mask", None)
                try:
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(self.device)
                        out = self.torch_model(input_ids, attention_mask)
                    else:
                        out = self.torch_model(input_ids)
                except TypeError:
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
        single = isinstance(out, dict)
        if single:
            t = out
            return {"combined": t, "combined_score": float(t["prob"])}
        else:
            res = []
            for item in out:
                res.append({"combined": item, "combined_score": float(item["prob"])})
            return res

if __name__ == "__main__":
    try:
        m = SentenceModel("combined")
        samples = ["I hate you", "Have a nice day"]
        print("Predict:", m.predict(samples))
    except Exception as e:
        print("Smoke test failed:", e)
