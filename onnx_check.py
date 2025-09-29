import onnxruntime as ort, os

p = r".\models\exports\model.onnx"
print("exists:", os.path.exists(p), "size:", os.path.getsize(p))

try:
    sess = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
    print("loaded OK")
    print("inputs:", [(i.name, i.shape, i.type) for i in sess.get_inputs()])
except Exception as e:
    print("onnx load error:", e)
