import onnxruntime as ort
import numpy as np
import joblib
from PIL import Image
from helper.loader import load_model

# === Konfigurasi ===
IMAGE_SIZE = (64, 64)
MAX_SEQ_LEN = 4

# === Tokenizer ===
TOKEN2IDX = joblib.load("./utils/token2idx.pkl")
IDX2TOKEN = joblib.load("./utils/idx2token.pkl")

# === Transformasi manual ===
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert('L')
    image = image.resize(IMAGE_SIZE)
    image_np = np.array(image, dtype=np.float32) / 255.0
    return image_np[np.newaxis, :, :]

# === Load ONNX model ===
onnx_session = load_model()

# === Fungsi Prediksi ===
def predict_from_images(images: list[Image.Image]) -> list[str]:
    tensors = [preprocess_image(img) for img in images]

    # Padding hingga MAX_SEQ_LEN
    while len(tensors) < MAX_SEQ_LEN:
        tensors.append(np.zeros_like(tensors[0]))

    # Shape akhir: [1, MAX_SEQ_LEN, 1, 64, 64]
    input_array = np.stack(tensors, axis=0)[np.newaxis, ...]

    # ONNX inference
    ort_inputs = {onnx_session.get_inputs()[0].name: input_array}
    ort_outs = onnx_session.run(None, ort_inputs)

    # Argmax prediksi
    predicted = np.argmax(ort_outs[0], axis=-1).squeeze(0).tolist()

    tokens = [
        IDX2TOKEN[idx]
        for idx in predicted
        if idx not in [TOKEN2IDX['<PAD>'], TOKEN2IDX['<SOS>'], TOKEN2IDX['<EOS>']]
    ]
    return tokens


