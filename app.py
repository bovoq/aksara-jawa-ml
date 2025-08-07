import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image, ImageOps
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import random
import joblib

# === Konfigurasi ===
IMAGE_SIZE = (64, 64)
MAX_SEQ_LEN = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Load Tokenizer dan Model ===
TOKEN2IDX = joblib.load("./utils/token2idx.pkl")
IDX2TOKEN = joblib.load("./utils/idx2token.pkl")
VOCAB_SIZE = len(TOKEN2IDX)

# === Model ===
class CNNEncoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        base = models.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, out_dim)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features.view(batch_size, seq_len, -1)

class DecoderLSTM(nn.Module):
    def __init__(self, cnn_out_dim, hidden_dim, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(cnn_out_dim, hidden_dim, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, vocab_size)

    def forward(self, encoder_outputs):
        lstm_out, _ = self.lstm(encoder_outputs)
        return self.classifier(lstm_out)

class AksaraModel(nn.Module):
    def __init__(self, cnn_out_dim=256, hidden_dim=256):
        super().__init__()
        self.encoder = CNNEncoder(cnn_out_dim)
        self.decoder = DecoderLSTM(cnn_out_dim, hidden_dim, VOCAB_SIZE)

    def forward(self, x):
        enc_out = self.encoder(x)
        return self.decoder(enc_out)

# === Load Model ===
model = AksaraModel()
model.load_state_dict(torch.load("./model/aksara_cnn_lstm.pth", map_location=DEVICE))
model.eval()

# === Transformasi Gambar ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

# === Soal Random (misal 20 soal)
SEMUA_AKSARA = list(TOKEN2IDX.keys())
SEMUA_AKSARA = [a for a in SEMUA_AKSARA if a not in ['<PAD>', '<SOS>', '<EOS>']]

def generate_random_questions(n=20):
    return [random.choices(SEMUA_AKSARA, k=random.randint(1, MAX_SEQ_LEN)) for _ in range(n)]

# === Session State Init ===
if "soal_list" not in st.session_state:
    st.session_state.soal_list = generate_random_questions()

if "soal_index" not in st.session_state:
    st.session_state.soal_index = 0

if "sequence_images" not in st.session_state:
    st.session_state.sequence_images = []

# === UI ===
st.title("✍️ Latihan Menulis Aksara Jawa")
st.markdown("Gambarlah aksara sesuai **soal di bawah ini** satu per satu, lalu tekan tombol Prediksi.")

# === Tampilkan Soal ===
soal = st.session_state.soal_list[st.session_state.soal_index]
st.subheader("📝 Soal")
st.info("Tuliskan: " + " - ".join(soal))

# === Layout: Centered canvas dan tombol ===
left, canvas_col, tombol_col, right = st.columns([1, 4, 2, 1])

with canvas_col:
    st.markdown("### ✍️ Gambar")
    canvas_result = st_canvas(
        fill_color="#000000",
        stroke_width=5,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=300,
        height=300,
        drawing_mode="freedraw",
        key="canvas",
    )

with tombol_col:
    st.markdown("###")
    if st.button("➕ Tambahkan"):
        if canvas_result.image_data is not None:
            img = Image.fromarray((canvas_result.image_data).astype("uint8")).convert("RGB")
            img = ImageOps.invert(img)
            img = img.resize((300, 300))
            st.session_state.sequence_images.append(img)
        else:
            st.warning("Belum ada gambar di canvas.")

    if st.button("♻️ Ulangi"):
        st.session_state.sequence_images = []

    if st.button("🗑️ Hapus Terakhir"):
        if st.session_state.sequence_images:
            st.session_state.sequence_images.pop()
        else:
            st.warning("Belum ada gambar untuk dihapus.")

    if st.button("🔀 Ganti Soal"):
        st.session_state.soal_index = (st.session_state.soal_index + 1) % len(st.session_state.soal_list)
        st.session_state.sequence_images = []

# === Preview Sequence ===
if st.session_state.sequence_images:
    st.subheader("🖼️ Gambar yang Ditulis")
    st.image(st.session_state.sequence_images, width=100)

# === Prediksi Model ===
if st.button("🔮 Prediksi"):
    imgs = st.session_state.sequence_images
    if len(imgs) == 0:
        st.warning("Kamu belum menambahkan gambar.")
    else:
        # Transform dan padding
        tensors = [transform(im) for im in imgs]
        while len(tensors) < MAX_SEQ_LEN:
            tensors.append(torch.zeros_like(tensors[0]))
        tensor_seq = torch.stack(tensors).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor_seq)
            predicted = output.argmax(dim=-1).squeeze(0).tolist()

        tokens = [IDX2TOKEN[idx] for idx in predicted if idx not in [TOKEN2IDX['<PAD>'], TOKEN2IDX['<SOS>'], TOKEN2IDX['<EOS>']]]
        st.success(f"🔤 Prediksi Model: **{' - '.join(tokens)}**")

        # Cek apakah cocok dengan soal
        if tokens == soal:
            st.balloons()
            st.success("✅ Jawaban BENAR!")
        else:
            st.error("❌ Jawaban masih salah, coba lagi.")