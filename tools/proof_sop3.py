import time
import os
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort

# ==========================================
#           FONT CONFIGURATION
# ==========================================
FONT_FAMILY = "Courier New"
FONT_SIZE_TITLE = 12
FONT_SIZE_AXIS_LABEL = 32
FONT_SIZE_TICK = 32
FONT_SIZE_LEGEND = 32
FONT_SIZE_ANNOTATION = 32

def main():
    # ==========================================
    #           1. SETUP & LOADING
    # ==========================================
    # Apply font family globally
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
    })
    onnx_path = "eye_state_mobilenet.onnx"
    
    if not os.path.exists(onnx_path):
        print(f"Error: Model file '{onnx_path}' not found.")
        return

    print(f"Loading ONNX model: {onnx_path}")

    # Prefer GPU if available, fall back to CPU
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    try:
        sess = ort.InferenceSession(onnx_path, providers=providers)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Inspect input to build the right dummy tensor
    inp = sess.get_inputs()[0]
    input_name = inp.name
    raw_shape = inp.shape

    # Replace dynamic dims (None or 'batch' strings) with 1 for benchmarking
    shape = [1 if (isinstance(d, str) or d is None) else d for d in raw_shape]
    
    # Create random dummy data of the correct shape (float32)
    dummy_input = np.random.randn(*shape).astype(np.float32)

    # ==========================================
    #           2. WARMUP
    # ==========================================
    # Wake up the execution provider so the first run isn't slow
    print("Warming up...")
    for _ in range(50):
        _ = sess.run(None, {input_name: dummy_input})

    # ==========================================
    #           3. BENCHMARK LOOP
    # ==========================================
    latencies = []
    iterations = 1000
    print(f"Running {iterations} inference tests...")

    for _ in range(iterations):
        start = time.perf_counter()
        _ = sess.run(None, {input_name: dummy_input})
        end = time.perf_counter()
        # Convert seconds to milliseconds
        latencies.append((end - start) * 1000)

    # ==========================================
    #           4. STATISTICS
    # ==========================================
    avg_lat = np.mean(latencies)
    min_lat = np.min(latencies)
    max_lat = np.max(latencies)
    p99_lat = np.percentile(latencies, 99) # 99th percentile (worst case)

    print("\n" + "="*30)
    print("   SOP 3 RESULTS")
    print("="*30)
    print(f"Average Latency: {avg_lat:.4f} ms")
    print(f"Min Latency:     {min_lat:.4f} ms")
    print(f"Max Latency:     {max_lat:.4f} ms")
    print(f"99% Percentile:  {p99_lat:.4f} ms")
    print(f"Input shape used: {shape}")

    # ==========================================
    #           5. VISUALIZATION (ZOOMED)
    # ==========================================
    plt.figure(figsize=(10, 5))
    
    # Create the boxplot
    # showfliers=False hides the extreme "background task" spikes
    plt.boxplot(
        latencies,
        vert=False,
        patch_artist=True,
        showfliers=False, 
        boxprops=dict(facecolor="#aec7e8", edgecolor="#1f77b4"),
        medianprops=dict(color="#d62728", linewidth=2),
    )

    # Add Average Line
    plt.axvline(avg_lat, color='red', linestyle='dashed', linewidth=2, label=f'Avg Speed: {avg_lat:.2f}ms')
    
    # --- DYNAMIC ZOOM LOGIC ---
    # Set the view limit based on the data, NOT the 33ms limit.
    # We look at the 99th percentile (approx 0.5-0.8ms) and give it some breathing room.
    # If p99 is 0.5ms, the graph will show up to ~0.75ms.
    view_limit = p99_lat * 1.5  
    plt.xlim(0, view_limit)

    # Add the "Off-Screen" Indicator for the 33ms limit
    # This draws an arrow pointing right to show the limit is way off the chart
    plt.text(
        view_limit * 0.95,  # X position (far right of the view)
        1.3,                # Y position (slightly above the box)
        "Real-Time Limit (33ms) →", 
        color='green', 
        fontweight='bold', 
        fontsize=FONT_SIZE_ANNOTATION,
        ha='right'          # Align text to the right
    )

    # Dynamic Title
    speedup = 33.33 / avg_lat
    plt.title(
        f"SOP 3: System Inference Latency\n(Model is {speedup:.0f}x faster than real-time requirement!)",
        fontsize=FONT_SIZE_TITLE,
    )
    plt.xlabel("Processing Time (ms)", fontsize=FONT_SIZE_AXIS_LABEL)
    plt.tick_params(axis='x', labelsize=FONT_SIZE_TICK)
    
    # Label the Y-axis clearly
    plt.yticks([1], [""], fontsize=FONT_SIZE_TICK) 
    
    plt.legend(loc="lower right", fontsize=FONT_SIZE_LEGEND)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    save_path = "SOP3_Latency_Zoomed.png"
    plt.savefig(save_path)
    print(f"\nGraph saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()