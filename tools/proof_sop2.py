import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import json

# FONT CONFIGURATION
FONT_FAMILY = "Courier New"
FONT_SIZE_AXIS_LABEL = 32
FONT_SIZE_TICK = 32
FONT_SIZE_LEGEND = 32
FONT_SIZE_ANNOTATION = 32

# CONFIG
LSTM_PATH = "blink_lstm.onnx"
LABEL_MAP_PATH = "lstm_label_map.json"

def main():
    # Apply font settings globally
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
    })

    # Load Label Map to find which index is "A" (.-) or "T" (-) or "E" (.)
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)
        # Find index for 'E' (Dot) and 'T' (Dash)
        idx_dot = label_map.get('E') 
        idx_dash = label_map.get('T') 

    sess = ort.InferenceSession(LSTM_PATH)
    
    # Test range: 0.05s to 1.0s
    durations = np.arange(0.05, 1.0, 0.01)
    dot_scores = []
    dash_scores = []
    
    print("Probing LSTM decision boundary...")
    
    for d in durations:
        # Create a sequence of ONE blink with duration 'd'
        # Shape: (1, 1, 1) -> (Batch, Time, Feature)
        dummy_input = np.array([[[d]]], dtype=np.float32)
        
        # Run inference
        outputs = sess.run(None, {'input': dummy_input})[0] # Logits
        
        # Softmax to get probability
        probs = np.exp(outputs) / np.sum(np.exp(outputs), axis=1, keepdims=True)
        
        dot_scores.append(probs[0][idx_dot])
        dash_scores.append(probs[0][idx_dash])

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(durations, dot_scores, label="Probability of 'E' (Dot)", color='blue')
    plt.plot(durations, dash_scores, label="Probability of 'T' (Dash)", color='red')
    
    # Find Intersection (The Learned Threshold)
    threshold_idx = np.argwhere(np.diff(np.sign(np.array(dot_scores) - np.array(dash_scores)))).flatten()
    if len(threshold_idx) > 0:
        learned_thresh = durations[threshold_idx[0]]
        plt.axvline(learned_thresh, color='green', linestyle='--', label=f"Learned Threshold: ~{learned_thresh:.2f}s")
        print(f"\n--- SOP 2 ANSWER ---")
        print(f"The LSTM learned that the boundary between Dot and Dash is approx {learned_thresh:.2f} seconds.")
    
    plt.xlabel("Blink Duration (seconds)", fontsize=FONT_SIZE_AXIS_LABEL)
    plt.ylabel("Model Confidence", fontsize=FONT_SIZE_AXIS_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICK)
    plt.yticks(fontsize=FONT_SIZE_TICK)
    plt.legend(fontsize=FONT_SIZE_LEGEND)
    plt.grid(True)
    plt.savefig("SOP2_Evidence_Threshold.png")
    print("Graph saved to SOP2_Evidence_Threshold.png")

    # --- ADDED THIS LINE TO OPEN THE INTERACTIVE WINDOW ---
    plt.show() 

if __name__ == "__main__":
    main()