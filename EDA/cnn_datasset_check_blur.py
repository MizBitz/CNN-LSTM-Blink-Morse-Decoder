import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
#           CONFIGURATION
# ==========================================
DATASET_DIR = r"S:\VSCode Projects\Backup Code\DATASET FINAL\DATASET FINAL\training_dataset"  # Your dataset folder
EXTENSIONS = ["*.jpg", "*.png", "*.jpeg"]
FONT_FAMILY = "Courier New"
FONT_SIZE = 32  # base size for axes/labels; adjust as desired
THRESHOLD_LABEL_FONT_SIZE = 24  # keep the blur-threshold label fixed

def calculate_sharpness_scores(root_dir):
    sharpness_scores = []
    image_paths = []
    
    print(f"Scanning {root_dir}...")
    
    # Recursively find all images
    files = []
    for ext in EXTENSIONS:
        files.extend(glob.glob(os.path.join(root_dir, "**", ext), recursive=True))
    
    print(f"Found {len(files)} images. Calculating Laplacian variance...")
    
    for filepath in files:
        # Read image
        img = cv2.imread(filepath)
        if img is None: continue
            
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # --- THE MAGIC METRIC ---
        # Calculate Variance of Laplacian
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        sharpness_scores.append(score)
        image_paths.append(filepath)

    return sharpness_scores, image_paths

def main():
    if not os.path.exists(DATASET_DIR):
        print(f"Error: {DATASET_DIR} not found.")
        return

    scores, paths = calculate_sharpness_scores(DATASET_DIR)
    
    if not scores:
        print("No images found.")
        return

    # Apply font configuration globally for the plot
    plt.rcParams.update({
        "font.family": FONT_FAMILY,
        "font.size": FONT_SIZE,
    })

# ==========================================
    #           VISUALIZATION
    # ==========================================
    plt.figure(figsize=(8, 6))
    plt.boxplot(scores, vert=True, patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="lightblue"))
    plt.xticks([1], ["Dataset Images"])
    plt.title("Distribution of Image Sharpness", fontsize=FONT_SIZE + 2)
    plt.ylabel("Laplacian Variance Score")
    plt.grid(axis='y', alpha=0.3)
    plt.gca().tick_params(axis="both", labelsize=FONT_SIZE)
    
    # Add the threshold line for context
    plt.axhline(y=45, color='red', linestyle='--', label='Blur Threshold (45)')
    plt.legend(fontsize=THRESHOLD_LABEL_FONT_SIZE)
    plt.show()

    # ==========================================
    #           STATISTICS
    # ==========================================
    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)
    
    print(f"\n--- Statistics ---")
    print(f"Total Images: {len(scores)}")
    print(f"Average Sharpness: {avg_score:.2f}")
    print(f"Min Sharpness: {min_score:.2f} (Blurriest)")
    print(f"Max Sharpness: {max_score:.2f} (Sharpest)")
    
    # Count images below threshold
    blurry_count = sum(1 for s in scores if s < 45)
    print(f"\nPotential Blurry Images (< 45): {blurry_count} ({blurry_count/len(scores)*100:.1f}%)")

if __name__ == "__main__":
    main()