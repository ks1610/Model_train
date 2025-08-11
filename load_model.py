import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import pickle
from torchvision import transforms

# ===== Model Classes (same as training) =====
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

# ===== Load Model & Label Encoder =====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

model = VisionTransformer(num_classes=len(le.classes_)).to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# ===== Preprocessing (same as training test_transform) =====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ===== Prediction Function =====
def predict_image(image_path):
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image_tensor)
        predicted_class_idx = outputs.argmax(dim=1).item()
        predicted_label = le.inverse_transform([predicted_class_idx])[0]
        print("Label classes:", list(le.classes_))
        print("Predicted label:", predicted_label)
    return predicted_label

# Example usage
image_path = r'D:\Trinh\AICam\test\test3.png' #đổi thành các ảnh trong file để test
prediction = predict_image(image_path)
print(f"Predicted class for image '{image_path}': {prediction}")
