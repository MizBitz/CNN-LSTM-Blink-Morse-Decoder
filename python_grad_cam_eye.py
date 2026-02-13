# grad_cam_eye.py
import torch, cv2, numpy as np
from PIL import Image
from torchvision import transforms
from train_cnn import get_model, IMAGE_SIZE  # reuse your model definition

MODEL_WEIGHTS = "eye_state_cnn.pth"   # or your latest .pth
IMAGE_PATH = r"S:\VSCode Projects\Backup Code\DATASET FINAL\DATASET FINAL\training_dataset\closed\closed_1033_64x64_dyn_L.jpg"           # eye patch or full face crop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device).eval()
state = torch.load(MODEL_WEIGHTS, map_location=device)
model.load_state_dict(state)

activations, gradients = [], []

def fwd_hook(_, __, out):
    activations.append(out.detach())

def bwd_hook(_, grad_in, grad_out):
    gradients.append(grad_out[0].detach())

preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

img_pil = Image.open(IMAGE_PATH).convert("RGB")
input_tensor = preprocess(img_pil).unsqueeze(0).to(device)

# Define layers to inspect:
# Layer 3: Low-level features (Edges, textures) - 8x8 resolution
# Layer 8: Mid-level features (Shapes, parts)   - 4x4 resolution
# Layer 12: High-level reasoning (Concepts)     - 2x2 resolution
layers_to_check = [3, 8, 12]

print(f"Processing layers: {layers_to_check}")

for i, layer_idx in enumerate(layers_to_check):
    target_layer = model.features[layer_idx]
    
    activations, gradients = [], []
    
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    
    # Forward + Backward
    model.zero_grad()
    logit = model(input_tensor).squeeze()
    logit.backward(retain_graph=True) # Retain graph for multiple backward passes if needed, though we actally re-forward here implicitly if we were doing this differently, but here we just need one pass per hook set? 
    # Actually, simplistic approach: just run the whole pipeline per layer to keep it clean.
    
    acts = activations[-1][0]
    grads = gradients[-1][0]
    
    weights = grads.mean(dim=(1, 2), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=0))
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam = cam.cpu().numpy()

    orig = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))
    overlay = cv2.addWeighted(orig, 0.5, heatmap, 0.5, 0)

    filename = f"gradcam_layer_{layer_idx}.jpg"
    cv2.imwrite(filename, overlay)
    print(f"Saved {filename} (Map size: {acts.shape[1]}x{acts.shape[2]})")

    h1.remove()
    h2.remove()

# Clean up
print("Done.")