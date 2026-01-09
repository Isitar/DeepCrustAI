import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.DeepCrustEnv import DeepCrustEnv

# --- CONFIGURATION ---
MODEL_V3_PATH = "deepcrust_fleet_v4_cold_pizza"  # 15 Inputs, No Normalization
MODEL_PRO_PATH = "deepcrust_pro_large"  # 18 Inputs, With Normalization
STATS_PRO_PATH = "deepcrust_pro_large_vecnormalize.pkl"
EPISODES = 10


def adapt_obs_for_v3(obs):
    # V3 expects 15 inputs. Env gives 18.
    # Remove Age columns (Indices 3, 7, 11)
    return np.delete(obs, [3, 7, 11])


def evaluate_v3():
    print(f"--- EVALUATING V3 (The Veteran) ---")
    env = DeepCrustEnv()

    try:
        model = PPO.load(MODEL_V3_PATH, device="cpu")
    except:
        print(f"❌ Error: Could not load {MODEL_V3_PATH}")
        return None

    stats = {"profit": [], "freshness": [], "backlog": []}

    for ep in range(EPISODES):
        obs, _ = env.reset(seed=42 + ep)  # Fixed seed for fairness
        done = False
        total_reward = 0
        deliveries = 0
        freshness_accum = 0
        backlog_accum = []

        while not done:
            # ADAPTER: 18 -> 15 inputs
            model_input = adapt_obs_for_v3(obs)

            action, _ = model.predict(model_input, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

            total_reward += reward
            backlog_accum.append(np.sum(env.orders))

            if reward > 0:
                deliveries += 1
                freshness_accum += reward

        stats["profit"].append(total_reward)
        stats["freshness"].append((freshness_accum / deliveries) if deliveries > 0 else 0)
        stats["backlog"].append(np.mean(backlog_accum))

    return {k: np.mean(v) for k, v in stats.items()}


def evaluate_pro():
    print(f"--- EVALUATING PRO (The New Heavyweight) ---")

    # 1. Setup Normalized Environment
    base_env = DummyVecEnv([lambda: DeepCrustEnv()])
    try:
        env = VecNormalize.load(STATS_PRO_PATH, base_env)
        env.training = False  # Don't update stats
        env.norm_reward = False  # We want REAL profit numbers
    except:
        print(f"❌ Error: Could not load {STATS_PRO_PATH}")
        return None

    try:
        model = PPO.load(MODEL_PRO_PATH, device="cpu")
    except:
        print(f"❌ Error: Could not load {MODEL_PRO_PATH}")
        return None

    stats = {"profit": [], "freshness": [], "backlog": []}

    for ep in range(EPISODES):
        # VecNormalize requires specific reset handling
        # We manually seed the internal env
        env.envs[0].reset(seed=42 + ep)
        obs = env.reset()

        done = False
        total_reward = 0
        deliveries = 0
        freshness_accum = 0
        backlog_accum = []

        while not done:
            # NO ADAPTER NEEDED (Env handles normalization automatically)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)

            # Note: VecEnv returns reward as an array [rew], get the scalar
            r = reward[0]

            total_reward += r
            # Access internal env to get orders/backlog
            backlog_accum.append(np.sum(env.envs[0].orders))

            if r > 0:
                deliveries += 1
                freshness_accum += r

        stats["profit"].append(total_reward)
        stats["freshness"].append((freshness_accum / deliveries) if deliveries > 0 else 0)
        stats["backlog"].append(np.mean(backlog_accum))

    return {k: np.mean(v) for k, v in stats.items()}


# --- RUN SHOWDOWN ---
stats_v3 = evaluate_v3()
stats_pro = evaluate_pro()

if stats_v3 and stats_pro:
    # --- PRINT TABLE ---
    df = pd.DataFrame([stats_v3, stats_pro], index=["V3 (Blind)", "Pro (Visible)"])
    print("\n--- FINAL SHOWDOWN RESULTS ---")
    print(df.round(2))

    # --- PLOT ---
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))
    colors = ['#7f8c8d', '#e74c3c']  # Grey vs Red

    metrics = [("profit", "Total Profit (Score)"),
               ("freshness", "Avg Quality per Pizza"),
               ("backlog", "Avg Angry Customers (Lower is Better)")]

    for i, (key, title) in enumerate(metrics):
        ax = axes[i]
        values = [stats_v3[key], stats_pro[key]]
        bars = ax.bar(["V3", "Pro"], values, color=colors)

        ax.set_title(title, fontsize=12, fontweight='bold', color='white')
        ax.grid(axis='y', linestyle='--', alpha=0.3)

        # Determine if Higher or Lower is better
        is_better = False
        if key == "backlog":
            is_better = values[1] < values[0]  # Lower is better
        else:
            is_better = values[1] > values[0]  # Higher is better

        # Add values on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height, f'{height:.1f}',
                    ha='center', va='bottom', color='white', fontweight='bold')

        # Add winner badge
        diff = ((values[1] - values[0]) / values[0]) * 100 if values[0] != 0 else 0
        txt_col = '#2ecc71' if is_better else '#e74c3c'
        ax.text(0.5, max(values) * 0.6, f"{diff:+.1f}%", ha='center', fontsize=16, color=txt_col, fontweight='bold')

    plt.suptitle("DeepCrust: Generation V3 vs. Pro Large", fontsize=16)
    plt.tight_layout()
    plt.savefig("final_showdown.png")
    print("\nGraph saved to final_showdown.png")
    plt.show()