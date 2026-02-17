import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Import your model definition
from train_lstm import BlinkLSTM

# ==========================================
#           CONFIGURATION
# ==========================================
MODEL_PATH = "blink_lstm.pth"
LABEL_MAP_PATH = "lstm_label_map.json"
DURATION_RANGE = (0.0, 1.5)  # Sweep from 0s to 1.5s
STEPS = 300                  # Number of points in the graph

# Reference means from your generation script (for visualization)
REF_DOT_MEAN = 0.20
REF_DASH_MEAN = 0.70

def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("❌ Error: Model or Label Map not found. Run prepare_lstm_data.py first.")
        return None, None

    # 1. Load Label Map
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    
    # Invert map to find 'E' and 'T' indices
    # 'E' is "." (Dot), 'T' is "-" (Dash)
    target_indices = {}
    if 'E' in label_map: target_indices['Dot'] = label_map['E']
    if 'T' in label_map: target_indices['Dash'] = label_map['T']
    
    if len(target_indices) < 2:
        print("❌ Error: Model must be trained on at least 'E' and 'T' to benchmark Dot/Dash.")
        return None, None

    # 2. Load Model
    device = torch.device("cpu") # CPU is fine for inference benchmark
    num_classes = len(label_map)
    
    # Initialize model with SAME hyperparameters as training
    model = BlinkLSTM(input_dim=1, hidden_dim=64, num_layers=2, num_classes=num_classes)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        return None, None

    return model, target_indices

def benchmark():
    model, targets = load_resources()
    if not model: return

    print(f"Benchmarking timing discrimination...")
    print(f"Tracking Class 'E' (Dot) vs Class 'T' (Dash)")

    # Generate input durations
    durations = np.linspace(DURATION_RANGE[0], DURATION_RANGE[1], STEPS)
    
    prob_dot = []
    prob_dash = []

    with torch.no_grad():
        for d in durations:
            # Prepare input: Batch=1, Seq_Len=1, Feature=1
            x = torch.tensor([[[d]]], dtype=torch.float32)
            lengths = torch.tensor([1], dtype=torch.long)
            
            # Forward pass
            logits = model(x, lengths)
            probs = F.softmax(logits, dim=1)
            
            # Extract specific probabilities
            p_dot = probs[0, targets['Dot']].item()
            p_dash = probs[0, targets['Dash']].item()
            
            prob_dot.append(p_dot)
            prob_dash.append(p_dash)

    # ==========================================
    #           VISUALIZATION
    # ==========================================
    plt.figure(figsize=(10, 6))
    
    # Plot Confidence Curves
    plt.plot(durations, prob_dot, label="Confidence: DOT ('E')", color='blue', linewidth=2.5)
    plt.plot(durations, prob_dash, label="Confidence: DASH ('T')", color='red', linewidth=2.5)

    # Find Crossover Point (Decision Boundary)
    crossover_idx = np.argwhere(np.diff(np.sign(np.array(prob_dot) - np.array(prob_dash)))).flatten()
    if len(crossover_idx) > 0:
        crossover_val = durations[crossover_idx[0]]
        plt.axvline(crossover_val, color='green', linestyle='--', alpha=0.7, label=f"Decision Boundary (~{crossover_val:.2f}s)")
        plt.text(crossover_val + 0.02, 0.5, f"{crossover_val:.2f}s", color='green', fontweight='bold')

    # Mark Ground Truth Means (from generate script)
    plt.axvline(REF_DOT_MEAN, color='blue', linestyle=':', alpha=0.4)
    plt.text(REF_DOT_MEAN, 0.95, "Ideal Dot", color='blue', ha='center', fontsize=9)
    
    plt.axvline(REF_DASH_MEAN, color='red', linestyle=':', alpha=0.4)
    plt.text(REF_DASH_MEAN, 0.95, "Ideal Dash", color='red', ha='center', fontsize=9)

    # Formatting
    plt.title("LSTM Timing Perception: Dot vs. Dash", fontsize=14)
    plt.xlabel("Input Duration (seconds)", fontsize=12)
    plt.ylabel("Model Confidence (Probability)", fontsize=12)
    plt.legend(loc="center right")
    plt.grid(True, alpha=0.3)
    plt.xlim(DURATION_RANGE)
    plt.ylim(0, 1.05)
    
    # Shaded Regions
    plt.fill_between(durations, 0, 1, where=(np.array(prob_dot) > np.array(prob_dash)), color='blue', alpha=0.05)
    plt.fill_between(durations, 0, 1, where=(np.array(prob_dash) > np.array(prob_dot)), color='red', alpha=0.05)

    output_file = "benchmark_timing.png"
    plt.savefig(output_file, dpi=300)
    print(f"✅ Graph saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    benchmark()