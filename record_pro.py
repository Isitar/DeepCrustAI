import matplotlib

# Force "Headless" mode. This fixes the "cannot reshape array" error
# by ensuring the window size is exactly what the code expects.
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.DeepCrustEnv import DeepCrustEnv
import os

# --- SETTINGS ---
MODEL_PATH = "deepcrust_pro_large"
STATS_PATH = "deepcrust_pro_large_vecnormalize.pkl"  # Crucial for the Pro model
OUTPUT_FILE = "deepcrust_pro_victory.gif"
STEPS = 300
FPS = 10


def record_pro_gif():
    print(f"--- PREPARING TO RECORD {MODEL_PATH} ---")

    # 1. Setup Environment with Normalization
    # The Pro model speaks a different "language" (normalized numbers),
    # so we MUST wrap the environment exactly how it was trained.

    # A. Create base environment wrapped in DummyVecEnv (Required for VecNormalize)
    # We use 'rgb_array' because 'human' can cause issues with the Agg backend
    base_env = DummyVecEnv([lambda: DeepCrustEnv(render_mode="rgb_array")])

    # B. Load the Normalization Statistics (The "Glasses")
    try:
        env = VecNormalize.load(STATS_PATH, base_env)
        env.training = False  # Don't update stats while recording
        env.norm_reward = False  # We want to see real rewards
    except FileNotFoundError:
        print(f"❌ Error: Could not find {STATS_PATH}.")
        print("Did you train the Pro model fully? The .pkl file is missing.")
        return

    # 2. Load the Pro Model
    try:
        model = PPO.load(MODEL_PATH)
        print(f"--- LOADED MODEL: {MODEL_PATH} ---")
    except:
        print(f"❌ Error: Could not load {MODEL_PATH}.")
        return

    # 3. Prepare for Recording
    frames = []
    print(f"🎥 Recording {STEPS} frames...")

    obs = env.reset()

    # ACCESS THE REAL ENV
    # env = VecNormalize -> env.venv = DummyVecEnv -> env.venv.envs[0] = DeepCrustEnv
    real_env = env.venv.envs[0]

    for i in range(STEPS):
        # Predict (Deterministic=True makes it show its best behavior)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        # Render Logic
        real_env.render()  # Updates the internal matplotlib figure

        # --- CAPTURE LOGIC (Your working method) ---
        real_env.fig.canvas.draw()

        # Get RGBA buffer
        image_rgba = np.array(real_env.fig.canvas.buffer_rgba())

        # Drop Alpha channel -> RGB
        image_rgb = image_rgba[:, :, :3]

        frames.append(image_rgb)
        # -------------------------------------------

        if i % 50 == 0:
            print(f"   ... captured step {i}/{STEPS}")

        # VecEnv resets automatically, but we check just in case logic changes
        # (We don't need manual reset code here usually)

    # 4. Save to GIF
    print(f"💾 Saving GIF to {OUTPUT_FILE}...")
    imageio.mimsave(OUTPUT_FILE, frames, fps=FPS, loop=0)
    print(f"✅ Done! Open '{OUTPUT_FILE}' to see the Pro model in action.")

    # Cleanup
    plt.close('all')
    real_env.close()


if __name__ == "__main__":
    record_pro_gif()