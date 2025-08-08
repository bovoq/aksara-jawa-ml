import os
import gdown
import onnxruntime as ort
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = "/tmp/aksara_model.onnx"
GDRIVE_FILE_ID = os.getenv("GDRIVE_FILE_ID")

os.environ["GDOWN_CACHE_DIR"] = "/tmp"

def load_model():
    if not os.path.exists(MODEL_PATH):
        print("🔽 Model not found. Downloading from Google Drive...")
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, use_cookies=False, quiet=False)
    else:
        print("✅ Model found in /tmp. Skipping download.")

    session = ort.InferenceSession(MODEL_PATH)
    return session
