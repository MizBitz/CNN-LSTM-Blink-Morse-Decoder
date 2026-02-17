import torch
import onnxruntime as ort
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ================= CONFIGURATION =================
MODEL_PATH = "eye_state_mobilenet.onnx" 
DATA_DIR = r"S:\VSCode Projects\Backup Code\cleaned_cnn_dataset" 
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)
# =================================================

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def evaluate_onnx(session, loader, title, save_matrix_name):
    print(f"\nRunning {title}...")
    
    all_preds = []
    all_labels = []
    input_name = session.get_inputs()[0].name
    
    for inputs, labels in loader:
        # Convert PyTorch tensor to Numpy for ONNX Runtime
        ort_inputs = {input_name: to_numpy(inputs)}
        ort_outs = session.run(None, ort_inputs)
        logits = ort_outs[0] 
        
        # Binary Logit Case (Logits > 0 means Open)
        preds = (logits > 0).astype(int).flatten()

        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    # Report
    target_names = ['closed', 'open']
    print(f"\n--- {title} Report ---")
    print(classification_report(all_labels, all_preds, target_names=target_names, digits=4))
    
    # Confusion Matrix Styling
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Courier New']})
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names, annot_kws={"size": 16})
    plt.title(f"{title}\nConfusion Matrix", fontsize=18)
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_matrix_name)
    print(f"✅ Matrix saved to {save_matrix_name}")

def main():
    print("Initializing SOP 1 & 4 Evaluation (ONNX Mode)...")

    try:
        # Load ONNX with available providers
        ort_session = ort.InferenceSession(MODEL_PATH, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f"Successfully loaded {MODEL_PATH}")
    except Exception as e:
        print(f"Error loading ONNX model: {e}")
        return

    # --- MATCHING YOUR TRAIN_CNN TRANSFORMS ---
    # Standard Inference/Baseline Transform
    transform_clean = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Robustness/Stress Transform (Applying random noise/affine to test SOP 4)
    transform_stress = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Setup Data Loaders
    ds_clean = datasets.ImageFolder(root=DATA_DIR, transform=transform_clean)
    loader_clean = DataLoader(ds_clean, batch_size=BATCH_SIZE, shuffle=False)
    
    ds_stress = datasets.ImageFolder(root=DATA_DIR, transform=transform_stress)
    loader_stress = DataLoader(ds_stress, batch_size=BATCH_SIZE, shuffle=False)

    # Run Evaluations
    evaluate_onnx(ort_session, loader_clean, "SOP 1: Baseline Accuracy", "SOP1_Baseline_Matrix.png")
    evaluate_onnx(ort_session, loader_stress, "SOP 4: Robustness Test", "SOP4_Robustness_Matrix.png")

    print("\nEvaluation Complete.")

if __name__ == "__main__":
    main()