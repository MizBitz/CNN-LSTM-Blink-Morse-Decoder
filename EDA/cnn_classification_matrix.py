import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
#           CONFIGURATION
# ==========================================
MODEL_PATH = "eye_state_cnn.pth"
DATA_DIR = r"S:\VSCode Projects\Backup Code\cleaned_cnn_dataset"  # Ensure this points to your dataset folder
BATCH_SIZE = 64
IMAGE_SIZE = (64, 64)

def get_model():
    """MobileNetV3-Small configured the same way as training (grayscale + binary head)."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # Replace first conv to accept 1 channel; initialize by averaging RGB weights.
    original_first = model.features[0][0]
    new_first = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        new_first.weight.copy_(original_first.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_first

    # Binary classifier head outputs a single logit (open vs closed).
    model.classifier[3] = nn.Linear(1024, 1)
    return model

# ==========================================
#           MAIN EXECUTION
# ==========================================
def main():
    print("Initializing SOP 1 Evaluation...")

    # 1. Setup Data
    # We use the same transforms as training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    try:
        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    except Exception as e:
        print(f"Error loading data: {e}")
        print(f"Make sure '{DATA_DIR}' exists.")
        return

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    
    try:
        # Prefer safe weight-only loading; fall back for older PyTorch versions.
        try:
            state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
        except TypeError:
            state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Successfully loaded {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Could not find {MODEL_PATH}. Make sure it's in the same folder.")
        return

    model.eval()
    
    # 3. Predict
    all_preds = []
    all_labels = []
    
    print("Running evaluation on dataset...")
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            logits = model(inputs).squeeze(1)
            preds = (logits > 0).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 4. Generate Report
    target_names = dataset.classes # Should be ['closed', 'open']
    print("\n" + "="*40)
    print("   SOP 1 ANSWER: CLASSIFICATION METRICS")
    print("="*40)
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # 5. Plot Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title("CNN Confusion Matrix (SOP 1 Evidence)")
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    save_path = "SOP1_Evidence_Matrix.png"
    plt.savefig(save_path)
    print(f"\n✅ Success! Matrix saved to {save_path}")
    print("You can now insert this image and the numbers above into Chapter 4.")

if __name__ == "__main__":
    main()