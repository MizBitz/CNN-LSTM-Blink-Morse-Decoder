# Train CNN - Final Optimized Version
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
import numpy as np
import matplotlib.pyplot as plt

# ================= CONFIGURATION =================
DATASET_DIR = r"S:\VSCode Projects\Backup Code\DATASET FINAL\DATASET FINAL\training_dataset"
MODEL_SAVE_PATH = "eye_state_mobilenet.onnx"
MODEL_PTH_PATH = "eye_state_mobilenet.pth"
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 50
IMAGE_SIZE = (64, 64) 
SEED = 42

# Training Hyperparameters
LABEL_SMOOTHING = 0.05 
AUGMENT_ENABLE = True
NUM_WORKERS = 2     # Speed up data loading (set to 0 if on Windows and getting errors)
PIN_MEMORY = True   # Speed up transfer to GPU

# =================================================

def save_training_plots(train_losses, val_losses):
    """Generates a combined learning curve for Training and Validation Loss."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Courier New', 'Courier'], 
        'font.size': 24,
        'axes.labelsize': 24,
        'axes.titlesize': 24,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'legend.fontsize': 24
    })
    
    epochs = range(1, len(train_losses) + 1)
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.plot(epochs, train_losses, color='blue', linewidth=3, label='Training Loss')
    ax.plot(epochs, val_losses, color='orange', linewidth=3, label='Validation Loss')
    
    ax.set_title('Learning Curve (Loss)')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss (CrossEntropy)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig('learning_curve.png')
    print("Graph saved as learning_curve.png")

def get_model():
    """Builds MobileNetV3 with a custom 16-neuron hidden layer."""
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Adapt first layer for Grayscale (1 channel)
    original_first_layer = model.features[0][0]
    new_first_layer = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        new_first_layer.weight.copy_(original_first_layer.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_first_layer
    
    # Custom Classifier: 1024 -> 16 -> 1
    model.classifier[3] = nn.Sequential(
        nn.Linear(1024, 16),  # Hidden Layer
        nn.ReLU(),            # Non-linear activation
        nn.Dropout(0.2),      # Extra regularization
        nn.Linear(16, 1)      # Output Layer
    )
    return model

def main():
    # 1. Setup & Reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    
    # 2. Dynamic Transform Construction
    # Base transforms (always applied)
    train_transform_steps = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE)
    ]
    
    # Augmentations (only if enabled)
    if AUGMENT_ENABLE:
        train_transform_steps.extend([
            transforms.RandomAffine(degrees=15, translate=(0.2, 0.2), scale=(0.85, 1.15)),
            transforms.RandomHorizontalFlip()
        ])
    
    # Final formatting
    train_transform_steps.extend([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_transform = transforms.Compose(train_transform_steps)
    
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 3. Dataset Loading
    try:
        train_full = datasets.ImageFolder(root=DATASET_DIR, transform=train_transform)
        test_full = datasets.ImageFolder(root=DATASET_DIR, transform=test_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # 4. Splitting & Sampling
    targets = np.array(train_full.targets)
    idx_closed, idx_open = np.where(targets == 0)[0], np.where(targets == 1)[0]
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx_closed)
    rng.shuffle(idx_open)

    t_ratio = 0.8
    n_c_t, n_o_t = int(len(idx_closed)*t_ratio), int(len(idx_open)*t_ratio)
    train_idx = np.concatenate([idx_closed[:n_c_t], idx_open[:n_o_t]])
    test_idx = np.concatenate([idx_closed[n_c_t:], idx_open[n_o_t:]])

    # Calculate class weights for sampling
    class_counts = np.bincount(targets[train_idx], minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[targets[train_idx]]
    
    # Optimized DataLoaders
    train_loader = DataLoader(
        Subset(train_full, train_idx.tolist()), 
        batch_size=BATCH_SIZE, 
        sampler=WeightedRandomSampler(torch.from_numpy(sample_weights), len(train_idx)),
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )
    
    test_loader = DataLoader(
        Subset(test_full, test_idx.tolist()), 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # 5. Model Initialization
    model = get_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_bal_acc': []}
    best_bal_acc = 0.0
    best_model_path = "best_" + MODEL_PTH_PATH

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        r_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device).float().unsqueeze(1)
            # Label Smoothing
            if LABEL_SMOOTHING > 0: 
                lbls = lbls * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()
            r_loss += loss.item()
        
        avg_train_loss = r_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        v_loss_sum = 0.0
        tp = tn = fp = fn = 0
        with torch.no_grad():
            for imgs, lbls in test_loader:
                imgs, lbls = imgs.to(device), lbls.to(device).float().unsqueeze(1)
                logits = model(imgs)
                v_loss_sum += criterion(logits, lbls).item()
                
                preds = (logits > 0).float()
                tp += int(((preds==1)&(lbls==1)).sum())
                tn += int(((preds==0)&(lbls==0)).sum())
                fp += int(((preds==1)&(lbls==0)).sum())
                fn += int(((preds==0)&(lbls==1)).sum())
        
        avg_val_loss = v_loss_sum / len(test_loader)
        
        # Metrics
        acc = 100.0 * (tp+tn)/max(tp+tn+fp+fn, 1)
        rec_open = tp / max(tp + fn, 1)
        rec_closed = tn / max(tn + fp, 1)
        bal_acc = 100.0 * 0.5 * (rec_open + rec_closed)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_acc'].append(acc)
        history['val_bal_acc'].append(bal_acc)

        # Save Best Model
        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            print(f"--> Epoch {epoch+1}: New Best BalAcc {best_bal_acc:.2f}%. Saving...")
            torch.save(model.state_dict(), best_model_path)

        scheduler.step()
        print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {avg_train_loss:.4f} | ValLoss: {avg_val_loss:.4f} | BalAcc: {bal_acc:.2f}%")

    # 6. Final Export Logic
    print(f"\nTraining Complete. Reloading Best Model (Acc: {best_bal_acc:.2f}%)...")
    
    # Fix: Use weights_only=True to suppress warnings
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    
    print(f"Saving Best weights to {MODEL_PTH_PATH} and exporting to ONNX...")
    torch.save(model.state_dict(), MODEL_PTH_PATH)
    
    # Export to ONNX (CPU Mode)
    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, 1, 64, 64)
    torch.onnx.export(
        model_cpu, 
        dummy_input, 
        MODEL_SAVE_PATH, 
        input_names=["input"], 
        output_names=["output"], 
        opset_version=17
    )
    
    save_training_plots(history['train_loss'], history['val_loss'])
    print("Done.")

if __name__ == "__main__":
    main()