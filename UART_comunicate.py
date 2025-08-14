import torch
import torch.nn as nn
from torchvision import transforms
import cv2
import serial
from PIL import Image
import numpy as np

# ===== Model Definition (must match your training) =====
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, emb_dim=256, img_size=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, emb_dim, H/patch, W/patch)
        x = x.flatten(2)  # (B, emb_dim, N_patches)
        x = x.transpose(1, 2)  # (B, N_patches, emb_dim)
        return x

class ViT(nn.Module):
    def __init__(self, num_classes=3, img_size=128, patch_size=16, emb_dim=256, depth=6, heads=8, mlp_dim=512, in_channels=3):
        super().__init__()
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 1 + self.patch_embedding.n_patches, emb_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        cls_output = x[:, 0]
        return self.fc(cls_output)

# ===== Device =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== Load Model =====
model = ViT(num_classes=3).to(device)  # adjust num_classes to your dataset
model.load_state_dict(torch.load("vit_model.pth", map_location=device))
model.eval()

# ===== Image Transform =====
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ===== UART Setup =====
uart = serial.Serial(port='COM3', baudrate=9600, timeout=1)

def send_uart_signal(message):
    uart.write(message.encode('utf-8'))
    print(f"UART sent: {message}")

# ===== Baseline Capture =====
cap = cv2.VideoCapture(0)
print("Capturing baseline environment... Please ensure no object is in view.")
ret, baseline = cap.read()
if not ret:
    print("Error: Unable to read from camera.")
    exit()
baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)

# ===== Difference Check =====
def is_significantly_different(frame, baseline_gray, threshold=30):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray, baseline_gray)
    return np.mean(diff) > threshold

# ===== Prediction Function =====
labels = ["cam", "chuoi", "other"]  # adjust to match your training labels

def predict_from_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img)
        _, predicted = torch.max(outputs, 1)
    return labels[predicted.item()]

# ===== Live Loop =====
print("Starting live prediction...")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if is_significantly_different(frame, baseline_gray):
        prediction = predict_from_frame(frame)
    else:
        prediction = "Default Environment"

    # Send UART if cam detected
    if prediction.lower() == "cam":
        send_uart_signal("1")

    cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
uart.close()
cv2.destroyAllWindows()
