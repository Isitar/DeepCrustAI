import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from env.DeepCrustEnv import DeepCrustEnv
from stable_baselines3 import PPO

# --- CONFIGURATION ---
MODEL_OLD = "deepcrust_fleet_v4_cold_pizza"  # 15 Inputs (Blind)
MODEL_NEW = "deepcrust_fleet_v5_visible_clock"  # 18 Inputs (Visible)
EPISODES = 10


def adapt_obs_for_old_model(obs):
    """
    New Obs (18 inputs):
    [T1_Data(3), T1_Age(1), T2_Data(3), T2_Age(1), T3_Data(3), T3_Age(1), Cities(5), Time(1)]

    Indices to remove:
    3  -> Truck 1 Age
    7  -> Truck 2 Age
    11 -> Truck 3 Age
    """
    return np.delete(obs, [3, 7, 11])


def evaluate_agent(model_name, label, use_adapter=False):
    env = DeepCrustEnv()  # Loads the correct, 18-input environment

    try:
        model = PPO.load(model_name, device="cpu")
    except:
        print(f"❌ Error: Could not load {model_name}")
        return None

    print(f"--- Simulating {label} ---")
    stats = {
        "rewards": [],
        "deliveries": [],
        "freshness_scores": []
    }

    for ep in range(EPISODES):
        obs, _ = env.reset(seed=42 + ep)
        done = False
        total_reward = 0
        deliveries = 0
        freshness_accum = 0

        while not done:
            # --- ADAPTER ---
            if use_adapter:
                # Strip the 3 age inputs so the old model sees the 15 inputs it knows
                model_input = adapt_obs_for_old_model(obs)
            else:
                model_input = obs
            # ---------------

            action, _ = model.predict(model_input, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

            total_reward += reward

            if reward > 0:
                deliveries += 1
                freshness_accum += reward

        stats["rewards"].append(total_reward)
        stats["deliveries"].append(deliveries)
        avg_freshness = (freshness_accum / deliveries) if deliveries > 0 else 0
        stats["freshness_scores"].append(avg_freshness)

    return {k: np.mean(v) for k, v in stats.items()}


# --- MAIN ---
# 1. Evaluate Old Model (Adapter ON)
stats_old = evaluate_agent(MODEL_OLD, "V4 (Blind)", use_adapter=True)

# 2. Evaluate New Model (Adapter OFF)
stats_new = evaluate_agent(MODEL_NEW, "V7 (Visible)", use_adapter=False)

if stats_old and stats_new:
    df = pd.DataFrame([stats_old, stats_new], index=["V4 (Blind)", "V7 (Visible)"])
    print("\n--- HEAD-TO-HEAD RESULTS ---")
    print(df.round(2))

    # Plot
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = ['#95a5a6', '#2ecc71']

    metrics = [("rewards", "Total Profit"),
               ("deliveries", "Pizzas Delivered"),
               ("freshness_scores", "Avg Quality (Reward/Pizza)")]

    for i, (key, title) in enumerate(metrics):
        ax = axes[i]
        values = [stats_old[key], stats_new[key]]
        bars = ax.bar(["V4 (Blind)", "V7 (Visible)"], values, color=colors)
        ax.set_title(title, fontsize=12, fontweight='bold', color='white')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        diff = values[1] - values[0]
        pct = (diff / values[0]) * 100 if values[0] != 0 else 0

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}',
                    ha='center', va='bottom', color='white', fontweight='bold')

        ax.text(0.5, max(values) * 0.5, f"{'+' if pct > 0 else ''}{pct:.1f}%",
                ha='center', fontsize=14, color='#2ecc71' if pct > 0 else 'white', fontweight='bold')

    plt.suptitle("DeepCrust Evolution: The Impact of Visible Freshness", fontsize=16)
    plt.tight_layout()
    plt.savefig("comparison_final.png")
    print("\nGraph saved to comparison_final.png")
    plt.show()