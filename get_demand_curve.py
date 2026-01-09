import matplotlib.pyplot as plt
import numpy as np
from env.DeepCrustEnv import DeepCrustEnv
from scipy.ndimage import gaussian_filter1d

# --- CONFIGURATION ---
STEPS = 200
SMOOTHING_SIGMA = 5  # Controls how "round" the curves are. Higher = smoother.

# Distinct, bright colors for dark background
COLORS = {
    1: '#e74c3c',  # Brugg Bhf (Red)
    2: '#3498db',  # Neumarkt (Blue)
    3: '#9b59b6',  # Königsfelden (Purple)
    4: '#f1c40f',  # Vindonissa (Yellow)
    5: '#2ecc71'  # Industrie (Green)
}


def get_demand_probability(t, node_idx):
    """
    Translates the env's _get_demand logic into a continuous probability score.
    This represents the 'intensity' of demand at any given moment.
    """
    # Base noise probability defined in the env (approx 10%)
    base_prob = 0.1

    added_prob = 0.0

    # Logic copied directly from DeepCrustEnv.py and translated to probability
    if node_idx == 1:  # Brugg Bhf
        if (20 < t < 50) or (160 < t < 190):
            added_prob = 0.6  # The "< 0.6" check

    elif node_idx == 2:  # Neumarkt
        if (80 < t < 120):
            added_prob = 0.5  # The "< 0.5" check

    elif node_idx == 5:  # Industrie
        if (t < 150):
            added_prob = 0.25  # The "< 0.25" check

    # Nodes 3 and 4 rely solely on base noise.

    # We combine them to get total 'demand intensity'
    return base_prob + added_prob


def generate_and_plot():
    env = DeepCrustEnv()  # Just to get node names

    x_axis = np.arange(STEPS)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(14, 8))

    # Iterate through cities 1 to 5
    for i in range(1, 6):
        city_name = env.node_names[i]
        color = COLORS[i]

        # 1. Generate the raw "rectangular" probability data
        raw_probs = [get_demand_probability(t, i) for t in x_axis]

        # 2. Apply Gaussian Smoothing to turn rectangles into curves
        # This creates the nice visual peak the user requested.
        smoothed_probs = gaussian_filter1d(raw_probs, sigma=SMOOTHING_SIGMA)

        # Ensure smoothing didn't dip below base noise too much at edges
        smoothed_probs = np.clip(smoothed_probs, 0.05, 1.0)

        # 3. Plot filled areas for a modern look
        ax.fill_between(x_axis, smoothed_probs, color=color, alpha=0.3)
        ax.plot(x_axis, smoothed_probs, color=color, linewidth=3, label=city_name)

    # --- Styling ---
    ax.set_title("FHNW DeepCrust: Predicted Demand Intensity (24h Cycle)",
                 fontsize=18, fontweight='bold', color='white', pad=20)

    ax.set_xlabel("Time of Day (Simulation Steps 0-200)", fontsize=14, fontweight='bold')
    ax.set_ylabel("Demand Probability / Intensity", fontsize=14, fontweight='bold')

    ax.set_ylim(0, 0.8)  # Set limit to frame the data nicely

    # Add time markers
    ax.axvline(x=20, color='white', linestyle='--', alpha=0.3)
    ax.text(20, 0.82, "Morning Rush Starts", color='white', alpha=0.7, ha='center', fontsize=9)

    ax.axvline(x=120, color='white', linestyle='--', alpha=0.3)
    ax.text(120, 0.82, "Lunch Ends", color='white', alpha=0.7, ha='center', fontsize=9)

    ax.axvline(x=160, color='white', linestyle='--', alpha=0.3)
    ax.text(160, 0.82, "Evening Rush Starts", color='white', alpha=0.7, ha='center', fontsize=9)

    ax.grid(color='#444444', linestyle=':', linewidth=0.5)

    # Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
              fancybox=True, shadow=True, ncol=5, facecolor='#222222', fontsize=12)

    plt.tight_layout()
    filename = "presentation_demand_smooth.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Graph saved to {filename}")
    plt.show()


if __name__ == "__main__":
    # Requires scipy: uv pip install scipy
    try:
        import scipy
    except ImportError:
        print("This script requires scipy for smoothing.")
        print("Please run: uv pip install scipy")
        exit()

    generate_and_plot()