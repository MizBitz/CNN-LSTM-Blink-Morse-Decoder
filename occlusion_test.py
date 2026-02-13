import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from train_cnn import get_model, IMAGE_SIZE
import matplotlib.pyplot as plt

# ================= CONFIG =================
MODEL_WEIGHTS = "eye_state_cnn.pth"
# Use the same image you were testing
IMAGE_PATH = r"S:\VSCode Projects\Backup Code\DATASET FINAL\DATASET FINAL\training_dataset\open\open_MEAD_2c6c4ff9_Josh_110096_64x64_dyn_L.jpg"
OCCLUSION_SIZE = 8   # Size of the gray block (8x8 pixels)
STRIDE = 2           # Step size (lower = higher res map, slower)
# ==========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device).eval()
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 1. Load and prep base image
img_pil = Image.open(IMAGE_PATH).convert("RGB")
# Get baseline probability for "Open"
input_tensor = preprocess(img_pil).unsqueeze(0).to(device)
with torch.no_grad():
    base_logit = model(input_tensor).item()
    base_prob = torch.sigmoid(torch.tensor(base_logit)).item()

print(f"Baseline 'Open' Probability: {base_prob:.4f}")

# 2. Occlusion Loop
width, height = img_pil.size
heatmap = np.zeros((height, width))

# Pad image slightly so occlusion can cover edges
pass

print("Running Occlusion Sensitivity Test (this may take a moment)...")

for y in range(0, height - OCCLUSION_SIZE + 1, STRIDE):
    for x in range(0, width - OCCLUSION_SIZE + 1, STRIDE):
        
        # Create occluded image
        img_occluded = img_pil.copy()
        draw = ImageDraw.Draw(img_occluded)
        # Draw gray rectangle (128 is mid-gray)
        draw.rectangle([x, y, x + OCCLUSION_SIZE, y + OCCLUSION_SIZE], fill=(128, 128, 128))
        
        # Inference
        input_tensor = preprocess(img_occluded).unsqueeze(0).to(device)
        with torch.no_grad():
            logit = model(input_tensor).item()
            prob = torch.sigmoid(torch.tensor(logit)).item()
        
        # heatmap value = How much did probability DROP?
        # If prob dropped a lot, this area was IMPORTANT.
        drop = max(0, base_prob - prob)
        
        # Fill the square in the heatmap
        heatmap[y:y+OCCLUSION_SIZE, x:x+OCCLUSION_SIZE] = drop

# 3. Normalize and Save
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
heatmap = np.uint8(255 * heatmap)

# Color map
heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
orig = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
orig = cv2.resize(orig, (width, height)) # Ensure match

# Overlay
overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)

cv2.imwrite("occlusion_sensitivity.jpg", overlay)
print("Saved occlusion_sensitivity.jpg. Blue = Unimportant, Red = Critical.")
