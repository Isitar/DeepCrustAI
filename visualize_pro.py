import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.DeepCrustEnv import DeepCrustEnv
import time

# --- CONFIGURATION ---
# Make sure these match exactly what you saved in train_pro.py
MODEL_PATH = "deepcrust_pro_large.zip"
STATS_PATH = "deepcrust_pro_large_vecnormalize.pkl"

def visualize():
    # 1. Create the base environment
    # We use DummyVecEnv because VecNormalize expects a vectorized environment
    env = DummyVecEnv([lambda: DeepCrustEnv(render_mode="human")])

    # 2. Load the "Translator" (Normalization Stats)
    # This ensures the model sees the exact same data range it saw during training
    try:
        env = VecNormalize.load(STATS_PATH, env)
    except FileNotFoundError:
        print(f"❌ Error: Could not find {STATS_PATH}.")
        print("Did you forget to save the .pkl file in train_pro.py?")
        return

    # 3. Disable Training Mode for Normalization
    # We don't want the test run to update the statistics, just use them.
    env.training = False
    env.norm_reward = False # We want to see the REAL rewards, not scaled ones

    # 4. Load the Brain
    print(f"--- LOADING PRO MODEL: {MODEL_PATH} ---")
    model = PPO.load(MODEL_PATH)

    print("--- STARTING VISUALIZATION ---")
    obs = env.reset()

    # Run loop
    for _ in range(1000):
        # We don't need to manually normalize 'obs' here.
        # The 'env' object (VecNormalize) handles it automatically!
        action, _ = model.predict(obs, deterministic=True)

        obs, reward, done, info = env.step(action)

        # Slow down slightly so humans can watch
        time.sleep(0.05)

        if done:
            obs = env.reset()

if __name__ == "__main__":
    visualize()