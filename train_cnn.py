# Train CNN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
import numpy as np

# ================= CONFIGURATION =================
DATASET_DIR = "training_dataset"
MODEL_SAVE_PATH = "eye_state_mobilenet.onnx"
MODEL_PTH_PATH = "eye_state_cnn.pth"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
EPOCHS = 10
IMAGE_SIZE = (64, 64) 
SEED = 42

LABEL_SMOOTHING = 0.05 

AUGMENT_ENABLE = True

# Blur augmentation
AUGMENT_BLUR_ENABLE = False
AUGMENT_BLUR_PROB = 0.25
AUGMENT_BLUR_KERNEL_SIZE = 3
AUGMENT_BLUR_SIGMA = (0.1, 1.0)

# Geometric augmentation
AUGMENT_AFFINE_ENABLE = True
AUGMENT_AFFINE_DEGREES = 15
AUGMENT_AFFINE_TRANSLATE = (0.20, 0.20)
AUGMENT_AFFINE_SCALE = (0.85, 1.15)
AUGMENT_AFFINE_SHEAR = 5
# =================================================

def get_model():
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    original_first_layer = model.features[0][0]
    new_first_layer = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
    with torch.no_grad():
        new_first_layer.weight.copy_(original_first_layer.weight.mean(dim=1, keepdim=True))
    model.features[0][0] = new_first_layer
    model.classifier[3] = nn.Linear(1024, 1) 
    return model

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    train_transform_steps = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
    ]
    if AUGMENT_ENABLE:
        if AUGMENT_BLUR_ENABLE and AUGMENT_BLUR_PROB > 0:
            train_transform_steps.append(
                transforms.RandomApply(
                    [
                        transforms.GaussianBlur(
                            kernel_size=AUGMENT_BLUR_KERNEL_SIZE,
                            sigma=AUGMENT_BLUR_SIGMA,
                        )
                    ],
                    p=float(AUGMENT_BLUR_PROB),
                )
            )
        if AUGMENT_AFFINE_ENABLE:
            train_transform_steps.append(
                transforms.RandomAffine(
                    degrees=AUGMENT_AFFINE_DEGREES,
                    translate=AUGMENT_AFFINE_TRANSLATE,
                    scale=AUGMENT_AFFINE_SCALE,
                    shear=AUGMENT_AFFINE_SHEAR,
                )
            )
        train_transform_steps.append(transforms.RandomHorizontalFlip(p=0.5))
    train_transform_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    train_transform = transforms.Compose(train_transform_steps)

    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    # Load Dataset
    try:
        train_full = datasets.ImageFolder(root=DATASET_DIR, transform=train_transform)
        test_full = datasets.ImageFolder(root=DATASET_DIR, transform=test_transform)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    print("class_to_idx:", train_full.class_to_idx)

    targets = np.array(train_full.targets)
    count_closed = int((targets == 0).sum())
    count_open = int((targets == 1).sum())
    print(f"Stats: {count_closed} Closed, {count_open} Open.")

    rng = np.random.default_rng(SEED)
    idx_closed = np.where(targets == 0)[0]
    idx_open = np.where(targets == 1)[0]
    rng.shuffle(idx_closed)
    rng.shuffle(idx_open)

    train_ratio = 0.8
    n_closed_train = int(len(idx_closed) * train_ratio)
    n_open_train = int(len(idx_open) * train_ratio)

    train_indices = np.concatenate([idx_closed[:n_closed_train], idx_open[:n_open_train]])
    test_indices = np.concatenate([idx_closed[n_closed_train:], idx_open[n_open_train:]])
    rng.shuffle(train_indices)
    rng.shuffle(test_indices)

    train_dataset = Subset(train_full, train_indices.tolist())
    test_dataset = Subset(test_full, test_indices.tolist())

    train_targets = targets[train_indices]
    class_counts = np.bincount(train_targets, minlength=2).astype(np.float32)
    class_weights = 1.0 / np.maximum(class_counts, 1.0)
    sample_weights = class_weights[train_targets]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights),
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = get_model().to(device)
    
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            if LABEL_SMOOTHING > 0:
                labels = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / max(len(train_loader), 1)

        model.eval()
        tp = tn = fp = fn = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device).float().unsqueeze(1)
                logits = model(images)
                predicted = (logits > 0).float()
                tp += int(((predicted == 1) & (labels == 1)).sum().item())
                tn += int(((predicted == 0) & (labels == 0)).sum().item())
                fp += int(((predicted == 1) & (labels == 0)).sum().item())
                fn += int(((predicted == 0) & (labels == 1)).sum().item())
        total = tp + tn + fp + fn
        acc = 100.0 * (tp + tn) / max(total, 1)

        rec_open = tp / max(tp + fn, 1)
        rec_closed = tn / max(tn + fp, 1)

        prec_open = tp / max(tp + fp, 1)
        prec_closed = tn / max(tn + fn, 1)

        bal_acc = 100.0 * 0.5 * (rec_open + rec_closed)

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}% | BalAcc: {bal_acc:.2f}% "
            f"| Open(P:{prec_open:.3f} R:{rec_open:.3f}) "
            f"| Closed(P:{prec_closed:.3f} R:{rec_closed:.3f}) "
            f"(TP:{tp} TN:{tn} FP:{fp} FN:{fn})"
        )

    print(f"Saving PyTorch weights to {MODEL_PTH_PATH}...")
    torch.save(model.state_dict(), MODEL_PTH_PATH)

    print(f"Saving ONNX to {MODEL_SAVE_PATH}...")
    model_cpu = model.to("cpu").eval()
    dummy_input = torch.randn(1, 1, 64, 64)
    torch.onnx.export(
        model_cpu,
        dummy_input,
        MODEL_SAVE_PATH,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )
    print("Done.")

if __name__ == "__main__":
    main()