import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from env.DeepCrustEnv import DeepCrustEnv
import os

# --- CONFIGURATION ---
# The model file you want to resume (without .zip)
OLD_MODEL_PATH = "deepcrust_fleet_v5_visible_clock"
# Where to save the smarter version
NEW_MODEL_PATH = "deepcrust_fleet_v5_visible_clock_longer"
EXTRA_STEPS = 1_000_000


def make_env(rank, seed=0):
    def _init():
        env = DeepCrustEnv()
        env.reset(seed=seed + rank)
        return env

    return _init


def continue_training():
    num_cpu = 16  # Ryzen 9 Power

    run = wandb.init(
        project="deepcrust-ai",
        config={"setup": "Extended Training Phase 2"},
        sync_tensorboard=True,
    )

    # 1. Recreate the exact same environment
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    print(f"--- LOADING BRAIN: {OLD_MODEL_PATH} ---")

    if not os.path.exists(f"{OLD_MODEL_PATH}.zip"):
        print(f"❌ Error: File {OLD_MODEL_PATH}.zip not found!")
        return

    # 2. Load the Model
    # We pass 'env' so the loaded model attaches to the new simulation
    model = PPO.load(OLD_MODEL_PATH, env=env, device="cuda")

    # OPTIONAL: Lower the learning rate slightly for fine-tuning
    # This helps it settle into the solution once it finds it.
    model.learning_rate = 0.0001

    print(f"--- RESUMING TRAINING (+{EXTRA_STEPS} Steps) ---")
    print("Watch for the 'Explained Variance' spike in WandB!")

    # 3. Train
    model.learn(
        total_timesteps=EXTRA_STEPS,
        reset_num_timesteps=False,  # Keep the graphs continuous
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )

    # 4. Save
    model.save(NEW_MODEL_PATH)
    print(f"✅ Success! Extended model saved to {NEW_MODEL_PATH}")
    run.finish()


if __name__ == "__main__":
    continue_training()