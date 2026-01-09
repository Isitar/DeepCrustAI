import matplotlib.pyplot as plt
import numpy as np
import imageio
from env.DeepCrustEnv import DeepCrustEnv
from stable_baselines3 import PPO
import os

# --- SETTINGS ---
MODEL_PATH = "deepcrust_fleet_v4_cold_pizza"
OUTPUT_FILE = "deepcrust_fleet_v4_cold.gif"
STEPS = 300
FPS = 10


def record_gif():
    # 1. Setup Environment
    # We use 'human' mode to ensure the window backend initializes correctly
    env = DeepCrustEnv(render_mode="human")
    obs, _ = env.reset(seed=42)

    # 2. Load Model
    try:
        model = PPO.load(MODEL_PATH)
        print(f"--- LOADED MODEL: {MODEL_PATH} ---")
    except:
        print(f"❌ Error: Could not load {MODEL_PATH}.")
        return

    frames = []
    print(f"🎥 Recording {STEPS} frames... (Do not close the popup window)")

    # 3. Run Simulation
    for i in range(STEPS):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, _, _ = env.step(action)

        # Render update
        env.render()

        # --- FIXED CAPTURE LOGIC (Matplotlib 3.8+ compatible) ---
        env.fig.canvas.draw()

        # Get RGBA buffer directly (returns a memoryview)
        # This works on all modern Matplotlib backends
        image_rgba = np.array(env.fig.canvas.buffer_rgba())

        # Drop the Alpha (transparency) channel to get just RGB
        image_rgb = image_rgba[:, :, :3]

        frames.append(image_rgb)
        # --------------------------------------------------------

        if i % 50 == 0:
            print(f"   ... captured step {i}/{STEPS}")

        if terminated:
            # Instead of stopping, we reset to keep the video smooth for 300 frames
            obs, _ = env.reset()

    # 4. Save to GIF
    print(f"💾 Saving GIF to {OUTPUT_FILE}...")
    imageio.mimsave(OUTPUT_FILE, frames, fps=FPS, loop=0)
    print(f"✅ Done! Open '{OUTPUT_FILE}' to watch your victory lap.")
    plt.close()


if __name__ == "__main__":
    record_gif()