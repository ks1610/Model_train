import os
import numpy as np
import pickle
from PIL import Image

# CONFIG
data_dir = 'D:\\Trinh\\AICam\\data'  # your folder path
image_size = (128, 128)    # must match model
classes = sorted(os.listdir(data_dir))  # assumes folder names are labels

X = []
y = []

for label in classes:
    class_dir = os.path.join(data_dir, label)
    for file_name in os.listdir(class_dir):
        file_path = os.path.join(class_dir, file_name)
        try:
            img = Image.open(file_path).convert('L')  # convert to grayscale
            img = img.resize(image_size)
            img_array = np.array(img).flatten()       # flatten to (16384,)
            X.append(img_array)
            y.append(label)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

X = np.array(X)
y = np.array(y)

print(f"Total samples: {len(X)}")
print(f"Labels: {set(y)}")

# Save to pickle
with open('data.pickle', 'wb') as f:
    pickle.dump({'X': X, 'y': y}, f)

print("Saved to data.pickle")
