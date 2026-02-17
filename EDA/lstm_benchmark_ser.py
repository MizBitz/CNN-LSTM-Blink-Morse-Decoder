import json, time
import numpy as np
import onnxruntime as ort
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

DATA_PATH = "morse_sequences.jsonl"
MODEL_PATH = "blink_lstm.onnx"
DOT_MAX = 0.3612
DASH_MIN = 0.5

# Morse lookup reused for confusion and SER on symbol strings
MORSE_TABLE = {
    ".-": "A", "-...": "B", "-.-.": "C", "-..": "D", ".": "E", "..-.": "F",
    "--.": "G", "....": "H", "..": "I", ".---": "J", "-.-": "K", ".-..": "L",
    "--": "M", "-.": "N", "---": "O", ".--.": "P", "--.-": "Q", ".-.": "R",
    "...": "S", "-": "T", "..-": "U", "...-": "V", ".--": "W", "-..-": "X",
    "-.--": "Y", "--..": "Z", "-----": "0", ".----": "1", "..---": "2", "...--": "3",
    "....-": "4", ".....": "5", "-....": "6", "--...": "7", "---..": "8", "----.": "9",
    ".-.-.-": ".", "--..--": ",", "..--..": "?", ".----.": "'", "-.-.--": "!", "-..-.": "/",
    "-.--.": "(", "-.--.-": ")", ".-...": "&", "---...": ":", "-.-.-.": ";", "-...-": "=",
    ".-.-.": "+", "-....-": "-", "..--.-": "_", ".-..-.": "\"", ".--.-.": "@"
}
CHAR_TO_MORSE = {v: k for k, v in MORSE_TABLE.items()}

def classify_duration(d):
    if d <= DOT_MAX: return "dot"
    if d >= DASH_MIN: return "dash"
    return None  # ambiguous gap

def levenshtein(a, b):
    if a == b: return 0
    if not a: return len(b)
    if not b: return len(a)
    prev = list(range(len(b)+1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            ins = curr[j-1] + 1
            dele = prev[j] + 1
            sub = prev[j-1] + (ca != cb)
            curr.append(min(ins, dele, sub))
        prev = curr
    return prev[-1]

def load_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

def main():
    rows = load_rows(DATA_PATH)
    if not rows:
        print("No data")
        return

    # Load label map to convert predicted indices into characters
    label_map_path = Path("lstm_label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    idx_to_label = {v: k for k, v in label_map.items()}

    sess = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider","CPUExecutionProvider"])
    inputs = {i.name: i for i in sess.get_inputs()}

    conf = Counter()
    ser_num = 0
    ser_den = 0
    latencies = []

    for row in rows:
        gt_label = row.get("label", "?")  # expected Morse token or class name
        durs = row.get("raw_durations", [])
        if not durs: continue

        arr = np.array([[min(d, 2.0) for d in durs]], dtype=np.float32)[:, :, None]
        feed = {"input": arr}
        if "lengths" in inputs:
            feed["lengths"] = np.array([arr.shape[1]], dtype=np.int64)

        t0 = time.perf_counter()
        logits = sess.run(None, feed)[0]
        t1 = time.perf_counter()
        latencies.append(t1 - t0)

        pred_idx = int(np.argmax(logits, axis=1)[0])
        pred_label = idx_to_label.get(pred_idx, "?")

        # Map predicted character to Morse for sequence-level comparison
        pred_morse = CHAR_TO_MORSE.get(pred_label, "")
        gt_morse = row.get("morse_seq", "") or ""

        # Dot/dash confusion at symbol level by aligning Morse strings
        max_len = max(len(gt_morse), len(pred_morse))
        for i in range(max_len):
            gt_sym = gt_morse[i] if i < len(gt_morse) else None
            pr_sym = pred_morse[i] if i < len(pred_morse) else None
            if gt_sym == ".":
                if pr_sym == ".": conf["TPd"] += 1
                elif pr_sym == "-": conf["FNd"] += 1
            elif gt_sym == "-":
                if pr_sym == "-": conf["TPD"] += 1
                elif pr_sym == ".": conf["FPd"] += 1

        # SER over Morse symbol strings
        ref = gt_morse
        hyp = pred_morse
        dist = levenshtein(hyp, ref)
        ser_num += dist
        ser_den += max(1, len(ref))

    # Metrics
    print("Confusion (dot/dash):", dict(conf))
    ser = ser_num / ser_den if ser_den else 0.0
    print(f"SER: {ser:.4f}  (num={ser_num}, den={ser_den})")
    if latencies:
        lat_arr = np.array(latencies)
        print(f"Latency: mean={lat_arr.mean()*1000:.2f} ms, p95={np.percentile(lat_arr,95)*1000:.2f} ms, n={len(lat_arr)}")

    # Plot/save confusion matrix as PNG
    cm = np.array([
        [conf.get("TPd", 0), conf.get("FNd", 0)],
        [conf.get("FPd", 0), conf.get("TPD", 0)],
    ])
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Pred Dot", "Pred Dash"])
    ax.set_yticks([0, 1], ["True Dot", "True Dash"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.05, pad=0.08)
    # Shift plot slightly left by widening right margin
    plt.subplots_adjust(left=0.23, right=0.88, top=0.94, bottom=0.12)
    out_path = Path(__file__).resolve().parent / "lstm_confusion.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Confusion matrix saved to {out_path}")

if __name__ == "__main__":
    main()