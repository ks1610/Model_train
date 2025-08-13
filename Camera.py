# import torch
# import torch.nn as nn
# from PIL import Image
# import numpy as np
# import pickle
# from torchvision import transforms
# import cv2
# import time

# # ===== Model Classes =====
# class PatchEmbedding(nn.Module):
#     def __init__(self, in_channels=1, patch_size=16, emb_dim=256, img_size=128):
#         super().__init__()
#         self.n_patches = (img_size // patch_size) ** 2
#         self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

#     def forward(self, x):
#         x = self.proj(x)
#         x = x.flatten(2)
#         x = x.transpose(1, 2)
#         return x

# class TransformerEncoderBlock(nn.Module):
#     def __init__(self, emb_dim, n_heads, mlp_ratio=4.0, dropout=0.2):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(emb_dim)
#         self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
#         self.norm2 = nn.LayerNorm(emb_dim)
#         self.mlp = nn.Sequential(
#             nn.Linear(emb_dim, int(mlp_ratio * emb_dim)),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(int(mlp_ratio * emb_dim), emb_dim),
#             nn.Dropout(dropout),
#         )

#     def forward(self, x):
#         x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
#         x = x + self.mlp(self.norm2(x))
#         return x

# class VisionTransformer(nn.Module):
#     def __init__(self, img_size=128, patch_size=16, in_channels=1, num_classes=10,
#                  emb_dim=256, depth=4, n_heads=4, mlp_ratio=4.0, dropout=0.2):
#         super().__init__()
#         self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
#         n_patches = self.patch_embed.n_patches
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
#         self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, emb_dim))
        
#         self.pos_drop = nn.Dropout(dropout)
#         self.blocks = nn.Sequential(*[
#             TransformerEncoderBlock(emb_dim, n_heads, mlp_ratio, dropout)
#             for _ in range(depth)
#         ])
#         self.norm = nn.LayerNorm(emb_dim)
#         self.head = nn.Linear(emb_dim, num_classes)

#     def forward(self, x):
#         B = x.shape[0]
#         x = self.patch_embed(x)
#         cls_tokens = self.cls_token.expand(B, -1, -1)
#         x = torch.cat((cls_tokens, x), dim=1)
#         x = self.pos_drop(x + self.pos_embed)
#         x = self.blocks(x)
#         x = self.norm(x)
#         return self.head(x[:, 0])

# # ===== Load Model =====
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# with open('label_encoder.pkl', 'rb') as f:
#     le = pickle.load(f)

# model = VisionTransformer(num_classes=len(le.classes_)).to(device)
# model.load_state_dict(torch.load('model.pth', map_location=device))
# model.eval()

# # ===== Preprocessing =====
# transform = transforms.Compose([
#     transforms.Grayscale(),
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.5,), (0.5,))
# ])

# # ===== Capture Empty Environment for 2–6 frames =====
# def capture_empty_environment(cap, min_frames=2, max_frames=6):
#     print(f"Capturing {min_frames}–{max_frames} frames of the empty environment...")
#     frames_captured = []
#     for i in range(min_frames, max_frames + 1):
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not capture frame.")
#             break
#         frames_captured.append(frame)
#         cv2.imshow("Empty Environment Capture", frame)
#         cv2.waitKey(500)  # Wait 0.5 seconds between captures
#     cv2.destroyWindow("Empty Environment Capture")
#     print("Empty environment capture complete.")
#     return frames_captured

# # ===== Prediction =====
# def predict_from_frame(frame):
#     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     img = Image.fromarray(img)
#     img_tensor = transform(img).unsqueeze(0).to(device)
#     with torch.no_grad():
#         outputs = model(img_tensor)
#         predicted_class_idx = outputs.argmax(dim=1).item()
#         return le.inverse_transform([predicted_class_idx])[0]

# # ===== Main =====
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open camera.")
#     exit()

# # Capture baseline empty environment images
# _ = capture_empty_environment(cap)

# print("Starting live prediction... Press 'q' to quit.")
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     prediction = predict_from_frame(frame)
#     cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     cv2.imshow("Camera Prediction", frame)
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pickle
from torchvision import transforms
import cv2
import time

# ===== Model Classes =====
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=1, patch_size=16, emb_dim=256, img_size=128):
        super().__init__()
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class TransformerEncoderBlock(nn.Module):
    def __init__(self, emb_dim, n_heads, mlp_ratio=4.0, dropout=0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, int(mlp_ratio * emb_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(mlp_ratio * emb_dim), emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=128, patch_size=16, in_channels=1, num_classes=10,
                 emb_dim=256, depth=4, n_heads=4, mlp_ratio=4.0, dropout=0.2):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, emb_dim, img_size)
        n_patches = self.patch_embed.n_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, emb_dim))
        
        self.pos_drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[
            TransformerEncoderBlock(emb_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_classes)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        return self.head(x[:, 0])

# ===== Load Model =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

model = VisionTransformer(num_classes=len(le.classes_)).to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# ===== Preprocessing =====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== Capture Empty Environment and Compute Baseline =====
def capture_empty_environment(cap, min_frames=2, max_frames=6):
    print(f"Capturing {min_frames}–{max_frames} frames of the empty environment...")
    frames_captured = []
    for _ in range(min_frames, max_frames + 1):
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break
        frames_captured.append(frame)
        cv2.imshow("Empty Environment Capture", frame)
        cv2.waitKey(500)  # Wait 0.5 seconds between captures
    cv2.destroyWindow("Empty Environment Capture")
    print("Empty environment capture complete.")

    # Compute average baseline in grayscale
    gray_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype("float32") for f in frames_captured]
    avg_baseline = np.mean(gray_frames, axis=0)
    return avg_baseline

# ===== Difference Detection =====
def is_significantly_different(frame, baseline, threshold=30):
    """Returns True if current frame is significantly different from the baseline."""
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype("float32")
    diff = cv2.absdiff(gray_frame, baseline)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

# ===== Prediction =====
def predict_from_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(img_tensor)
        predicted_class_idx = outputs.argmax(dim=1).item()
        return le.inverse_transform([predicted_class_idx])[0]

# ===== Main =====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Capture baseline empty environment images
baseline = capture_empty_environment(cap)

print("Starting live prediction... Ctrl+c to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if is_significantly_different(frame, baseline):
        prediction = predict_from_frame(frame)
    else:
        prediction = "Default Environment"

    cv2.putText(frame, f"Prediction: {prediction}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Camera Prediction", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
