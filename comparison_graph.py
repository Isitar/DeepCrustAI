import matplotlib.pyplot as plt
import numpy as np
from env.DeepCrustEnv import DeepCrustEnv
from stable_baselines3 import PPO

# --- SETTINGS ---
steps = 200  # Length of one 'day'
model_path_1 = "deepcrust_fleet_v2_parallel"  # Use your latest model filename
model_path_2 = "deepcrust_fleet_v2_parallel"  # Use your latest model filename


def run_episode(agent_type, model=None):
    env = DeepCrustEnv()  # No render for this, just data
    obs, _ = env.reset(seed=42)  # Same seed for fairness!

    cumulative_reward = 0
    history = []

    for _ in range(steps):
        if agent_type == "Random":
            action = env.action_space.sample()
        else:
            action, _ = model.predict(obs, deterministic=True)

        obs, reward, terminated, _, _ = env.step(action)

        # Accumulate reward to show the "Growth" over time
        cumulative_reward += reward
        history.append(cumulative_reward)

        if terminated:
            break

    return history


# 1. Run Random Agent
print("Running Random Agent...")
ai1_history = run_episode("Random")

# print("Running Trained AI...")
# if str(model_path_1).endswith(".zip"): model_path = model_path_1[:-4]  # Clean path
# model = PPO.load(model_path_1)
# ai1_history = run_episode("AI", model)
ai1_name = "Random"

# 2. Run AI Agent
print("Running Trained AI...")
if str(model_path_2).endswith(".zip"): model_path = model_path_2[:-4]  # Clean path
model = PPO.load(model_path_2)
ai2_history = run_episode("AI", model)
ai2_name = "DeepCrust V1"

# 3. Plot
plt.style.use('dark_background')  # Looks cooler
plt.figure(figsize=(10, 6))

plt.plot(ai1_history, color='#e74c3c', linewidth=2, linestyle='--', label=f'{ai1_name} (Final: {ai1_history[-1]:.1f})')
plt.plot(ai2_history, color='#2ecc71', linewidth=3, label=f'{ai2_name} (Final: {ai2_history[-1]:.1f})')

plt.title(f'Performance Comparison: {ai1_name} vs {ai2_name}', fontsize=16, color='white')
plt.xlabel('Time Steps (0-200)', fontsize=12)
plt.ylabel('Cumulative Profit', fontsize=12)
plt.grid(color='#444444', linestyle=':', linewidth=0.5)
plt.legend(fontsize=12)

# Save it for your slides
plt.savefig("comparison_chart.png", dpi=150)
print("Graph saved to comparison_chart.png")
plt.show()
