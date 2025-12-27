import matplotlib.pyplot as plt
import numpy as np
from env.DeepCrustEnv import DeepCrustEnv
from stable_baselines3 import PPO

# --- CONFIGURATION ---
STEPS = 200
SEED = 42
CITY_NAMES = ["Brugg Bhf", "Neumarkt", "Königsfelden", "Vindonissa", "Industrie"]


def get_orders(env):
    """Helper to get orders regardless of env version."""
    if hasattr(env, 'orders'):
        return env.orders.copy()
    start = env.n_trucks * 2
    end = start + env.n_cities
    return env.state[start:end].copy()


def get_demand_curves(agent_type="none", model_path=""):
    env = DeepCrustEnv()
    # CRITICAL: Same seed ensures exact same customer orders for fair comparison
    obs, _ = env.reset(seed=SEED)

    model = None
    if agent_type == "ai":
        try:
            model = PPO.load(model_path)
        except:
            print(f"Warning: Model {model_path} not found.")
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

print("2. Simulating 'AI 1' (Yellow)...")
ai1_name="random"
data_ai1 = get_demand_curves(agent_type="random")

print("3. Simulating 'AI 2' (Green)...")
ai2_name="Deep Crust AI1"
data_ai2 = get_demand_curves(agent_type="ai", model_path="deepcrust_fleet_v2_parallel.zip")

# --- PLOTTING ---
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i in range(5):
    ax = axes[i]
    city_name = CITY_NAMES[i]

    # Get data for this specific city
    y_none = data_none[i]
    y_ai1 = data_ai1[i]
    y_ai2 = data_ai2[i]

    # PLOT LINES
    # 1. Total Demand (Baseline)
    ax.plot(y_none, color='#e74c3c', linestyle='--', linewidth=2, label='Incoming Demand (Baseline)')

    # 2. AI 1
    ax.plot(y_ai1, color='#f1c40f', linewidth=2, alpha=0.8, label=ai1_name)

    # 3. AI Agent
    ax.plot(y_ai2, color='#2ecc71', linewidth=3, label=ai2_name)

    ax.fill_between(range(len(y_ai2)), y_ai1, y_ai2, where=(np.array(y_ai2) < np.array(y_ai1)),
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
