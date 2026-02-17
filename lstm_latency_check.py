import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

# Import your model definition
from train_lstm import BlinkLSTM

# ==========================================
#           CONFIGURATION
# ==========================================
MODEL_PATH = "blink_lstm.pth"
LABEL_MAP_PATH = "lstm_label_map.json"
ITERATIONS = 1000
SEQ_LENGTH = 5  # Simulate a 5-blink character

# --- GRAPH STYLING (MATCHING SOP 3) ---
FONT_FAMILY = "Courier New"
FONT_SIZE_TITLE = 12
FONT_SIZE_AXIS_LABEL = 32
FONT_SIZE_TICK = 32
FONT_SIZE_LEGEND = 32
FONT_SIZE_ANNOTATION = 32

def load_resources():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("❌ Error: Model or Label Map not found.")
        return None, None

    # Load Label Map
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
    num_classes = len(label_map)

    # Load Model (CPU is preferred for latency benchmarking to show 'worst case')
    device = torch.device("cpu")
    model = BlinkLSTM(input_dim=1, hidden_dim=64, num_layers=2, num_classes=num_classes).to(device)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
        model.eval()
        return model, device
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        return None, None

def main():
    # 1. APPLY FONTS GLOBALLY
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
    })

    # 2. LOAD MODEL
    model, device = load_resources()
    if not model: return

    print(f"Running LSTM Latency Benchmark on {device.type.upper()}...")
    
    # 3. PREPARE DUMMY INPUT
    # Shape: (Batch=1, Seq=5, Features=1)
    dummy_input = torch.randn(1, SEQ_LENGTH, 1).to(device)
    dummy_lengths = torch.tensor([SEQ_LENGTH], dtype=torch.long).to(device)

    # 4. WARMUP
    print("Warming up...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(dummy_input, dummy_lengths)

    # 5. BENCHMARK LOOP
    latencies = []
    print(f"Running {ITERATIONS} inference tests...")
    
    with torch.no_grad():
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            _ = model(dummy_input, dummy_lengths)
            end = time.perf_counter()
            latencies.append((end - start) * 1000) # Convert to ms

    # 6. STATISTICS
    avg_lat = np.mean(latencies)
    p99_lat = np.percentile(latencies, 99)
    print(f"Average Latency: {avg_lat:.4f} ms")
    print(f"99% Percentile:  {p99_lat:.4f} ms")

    # ==========================================
    #           VISUALIZATION
    # ==========================================
    plt.figure(figsize=(10, 5))

    # Create Boxplot
    plt.boxplot(
        latencies,
        vert=False,
        patch_artist=True,
        showfliers=False,  # Hide extreme outliers
        boxprops=dict(facecolor="#aec7e8", edgecolor="#1f77b4"),
        medianprops=dict(color="#d62728", linewidth=2),
    )

    # Add Average Line
    plt.axvline(avg_lat, color='red', linestyle='dashed', linewidth=2, label=f'Avg Speed: {avg_lat:.2f}ms')

    # --- DYNAMIC ZOOM LOGIC ---
    # Zoom in to show the distribution clearly, ignoring the empty space up to 33ms
    view_limit = max(p99_lat * 1.5, avg_lat * 2)
    plt.xlim(0, view_limit)

    # Add "Real-Time Limit" Arrow (Pointing off-screen if it's far away)
    if view_limit < 33:
        plt.text(
            view_limit * 0.98, 
            1.3, 
            "Real-Time Limit (33ms) →", 
            color='green', 
            fontweight='bold', 
            fontsize=FONT_SIZE_ANNOTATION,
            ha='right'
        )
    else:
        plt.axvline(33, color='green', linestyle=':', linewidth=2)
        plt.text(33, 1.3, " Real-Time Limit", color='green', fontweight='bold')

    # Dynamic Title
    speedup = 33.33 / avg_lat
    plt.title(
        f"LSTM Inference Latency\n(Model is {speedup:.0f}x faster than real-time requirement!)",
        fontsize=FONT_SIZE_TITLE,
    )
    plt.xlabel("Processing Time (ms)", fontsize=FONT_SIZE_AXIS_LABEL)
    plt.tick_params(axis='both', labelsize=FONT_SIZE_TICK)
    
    # Clean Y-axis
    plt.yticks([1], [""], fontsize=FONT_SIZE_TICK)

    plt.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    output_file = "benchmark_lstm_latency.png"
    plt.savefig(output_file, dpi=300)
    print(f"\nGraph saved to {output_file}")
    plt.show()

if __name__ == "__main__":
    main()