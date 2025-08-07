import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import joblib

# === Konfigurasi ===
IMAGE_SIZE = (64, 64)
MAX_SEQ_LEN = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Tokenizer ===
TOKEN2IDX = joblib.load("./utils/token2idx.pkl")
IDX2TOKEN = joblib.load("./utils/idx2token.pkl")
VOCAB_SIZE = len(TOKEN2IDX)

# === Transformasi ===
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(),
    transforms.ToTensor(),
])

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
model.to(DEVICE)
model.eval()

def predict_from_images(images: list[Image.Image]) -> list[str]:
    tensors = [transform(img) for img in images]
    while len(tensors) < MAX_SEQ_LEN:
        tensors.append(torch.zeros_like(tensors[0]))

    tensor_seq = torch.stack(tensors).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(tensor_seq)
        predicted = output.argmax(dim=-1).squeeze(0).tolist()

    tokens = [IDX2TOKEN[idx] for idx in predicted if idx not in [TOKEN2IDX['<PAD>'], TOKEN2IDX['<SOS>'], TOKEN2IDX['<EOS>']]]
    return tokens
