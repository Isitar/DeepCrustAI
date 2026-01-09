import gymnasium as gym
from stable_baselines3 import PPO
from env.DeepCrustEnv import DeepCrustEnv
import matplotlib.pyplot as plt
import os

env = DeepCrustEnv(render_mode="human")
obs, _ = env.reset()

# 2. Load the Model you just trained
model_name = "deepcrust_pro_large"

print(f"--- LOADING MODEL: {model_name} ---")
if not os.path.exists(f"{model_name}.zip"):
    print(f"ERROR: Could not find {model_name}.zip. Check if train.py finished successfully.")
    exit()

model = PPO.load(model_name)

print("--- VISUALIZING TRAINED FLEET ---")
print("Watch Node 1 (Uni) and Node 3 (Downtown).")

total_reward = 0
done = False

# Run for one full 'day' (200 steps) or until you close it
while not done:
    # Predict the action (Deterministic = Best possible move)
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, _ = env.step(action)
    total_reward += reward

    # Check if simulation ended
    done = terminated or truncated

print(f"Simulation finished. Total Reward: {total_reward:.2f}")

# Keep window open
print("Close the graph window to exit.")
plt.ioff()
plt.show()