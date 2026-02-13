import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ==========================================
#  USER CONFIGURATION
# ==========================================

# 1. FONT SETTINGS
# We set the global font family to monospace, and specifically 'Courier New'.
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.monospace'] = ['Courier New', 'Courier', 'DejaVu Sans Mono']

# 2. FONT SIZES
FONTS = {
    'title': 26,       # Title above each matrix
    'axis_label': 28,  # "True Label" / "Predicted Label"
    'tick_label': 28,  # "open" / "closed"
    'annot': 24,       # The numbers inside the boxes
    'vs_text': 28      # The "VS" text size
}

# 3. TITLES
TITLE_LEFT = "Ideal"
TITLE_RIGHT = "Augmented"

# 4. COLORS
# Suggestions: 'Blues', 'Greens', 'Reds', 'Purples', 'YlGnBu'
COLOR_MAP = 'Blues' 

# ==========================================

def get_formatted_labels(data):
    """
    Creates a text array with comma separators (e.g., "13,248")
    for easier reading.
    """
    return np.array([["{:,}".format(val) for val in row] for row in data])

def plot_styled_matrix(ax, data, labels, title):
    """
    Plots a single styled confusion matrix using the global font settings.
    """
    annot_labels = get_formatted_labels(data)

    sns.heatmap(
        data,
        annot=annot_labels,     # Use custom labels with commas
        fmt='',                 # vital: tell seaborn to use the string array as-is
        cmap=COLOR_MAP,
        cbar=False,             # Turn off colorbar for a cleaner look
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        square=True,
        linewidths=2,           # Gap between cells
        linecolor='white',      # White lines between cells
        annot_kws={"size": FONTS['annot'], "weight": "bold"}
    )
    
    # Titles and Labels
    ax.set_title(title, fontsize=FONTS['title'], pad=20, weight='bold')
    ax.set_ylabel('True Label', fontsize=FONTS['axis_label'], labelpad=10)
    ax.set_xlabel('Predicted Label', fontsize=FONTS['axis_label'], labelpad=10)
    
    # Tick formatting
    ax.tick_params(axis='both', which='major', labelsize=FONTS['tick_label'], length=0)


def create_courier_comparison():
    # --- Data Definition ---
    # Left Matrix Data
    cm_left = np.array([
        [13248, 3], 
        [126, 39667]
    ])
    
    # Right Matrix Data
    cm_right = np.array([
        [13236, 15], 
        [253, 39540]
    ])
    
    class_names = ['closed', 'open']

    # --- Setup Figure ---
    # Width ratios: 1 part left, 0.2 parts gap (for VS), 1 part right
    fig, axes = plt.subplots(1, 3, figsize=(16, 7), gridspec_kw={'width_ratios': [1, 0.2, 1]})
    
    # Set a white background
    fig.patch.set_facecolor('white')

    # --- Plot Left ---
    plot_styled_matrix(axes[0], cm_left, class_names, TITLE_LEFT)

    # --- Plot Center (VS) ---
    axes[1].axis('off')
    # Placing the VS text
    axes[1].text(0.5, 0.5, 'VS', 
                 ha='center', va='center', 
                 fontsize=FONTS['vs_text'], 
                 weight='bold', 
                 color='#333333') # Dark grey

    # --- Plot Right ---
    plot_styled_matrix(axes[2], cm_right, class_names, TITLE_RIGHT)

    # --- Final Layout ---
    plt.tight_layout()
    
    # Save
    filename = 'confusion_matrix_courier.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Image saved as '{filename}'")
    # plt.show() # Uncomment to view immediately

if __name__ == "__main__":
    create_courier_comparison()