import matplotlib.pyplot as plt
import numpy as np
from env.DeepCrustEnv import DeepCrustEnv
from stable_baselines3 import PPO

# --- CONFIGURATION ---
MODEL_PATH = "deepcrust_fleet_v2_parallel.zip"
STEPS = 200
SEED = 42
CITY_NAMES = ["University", "Burbs", "Downtown", "Stadium", "Industrial"]


def get_orders(env):
    """Helper to get orders regardless of env version."""
    if hasattr(env, 'orders'):
        return env.orders.copy()
    start = env.n_trucks * 2
    end = start + env.n_cities
    return env.state[start:end].copy()


def get_demand_curves(agent_type="none"):
    env = DeepCrustEnv()
    # CRITICAL: Same seed ensures exact same customer orders for fair comparison
    obs, _ = env.reset(seed=SEED)

    model = None
    if agent_type == "ai":
        try:
            model = PPO.load(MODEL_PATH)
        except:
            print(f"Warning: Model {MODEL_PATH} not found.")
            return [[] for _ in range(5)]

    histories = [[] for _ in range(5)]

    for _ in range(STEPS):
        # --- DECISION MAKING ---
        if agent_type == "ai" and model:
            action, _ = model.predict(obs, deterministic=True)
        elif agent_type == "random":
            action = env.action_space.sample()  # Random moves
        else:
            action = [0] * env.n_trucks  # Sit at Hub (No Delivery)

        obs, _, done, _, _ = env.step(action)

        current_orders = get_orders(env)
        for i in range(5):
            histories[i].append(current_orders[i])

        if done:
            break

    return histories


# --- MAIN EXECUTION ---
print("1. Simulating 'No Delivery' (Red)...")
data_none = get_demand_curves(agent_type="none")

print("2. Simulating 'Random Agent' (Yellow)...")
data_random = get_demand_curves(agent_type="random")

print("3. Simulating 'AI Agent' (Green)...")
data_ai = get_demand_curves(agent_type="ai")

if not data_ai[0]:
    print("Error: No AI data generated.")
    exit()

# --- PLOTTING ---
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i in range(5):
    ax = axes[i]
    city_name = CITY_NAMES[i]

    # Get data for this specific city
    y_none = data_none[i]
    y_random = data_random[i]
    y_ai = data_ai[i]

    # PLOT LINES
    # 1. Total Demand (Baseline)
    ax.plot(y_none, color='#e74c3c', linestyle='--', linewidth=2, label='Incoming Demand (Baseline)')

    # 2. Random Agent
    ax.plot(y_random, color='#f1c40f', linewidth=2, alpha=0.8, label='Random Agent')

    # 3. AI Agent
    ax.plot(y_ai, color='#2ecc71', linewidth=3, label='DeepCrust AI')

    # Fill area to show AI improvement over Random
    # (Optional: Fills green where AI is lower than Random)
    ax.fill_between(range(len(y_ai)), y_random, y_ai, where=(np.array(y_ai) < np.array(y_random)),
                    color='#2ecc71', alpha=0.1, interpolate=True)

    ax.set_title(f"{city_name}", fontsize=14, color='white', fontweight='bold')
    ax.grid(color='#444444', linestyle=':', linewidth=0.5)

    # Legend only on first graph
    if i == 0:
        ax.legend(loc='upper left', frameon=True, facecolor='#222222')

# Hide empty subplot
axes[5].axis('off')

plt.suptitle(f"Fleet Benchmark: AI vs Random vs Baseline", fontsize=18, color='white')
plt.tight_layout()
plt.savefig("demand_comparison_full.png")
print("Graph saved to demand_comparison_full.png")
plt.show()