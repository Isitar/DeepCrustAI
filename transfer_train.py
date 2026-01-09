import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from env.DeepCrustEnv import DeepCrustEnv
import os

# --- CONFIGURATION ---
# 1. The "Smart but ignorant" model (V3)
PRETRAINED_MODEL = "deepcrust_fleet_v2_parallel"

# 2. The new filename for the adapted model
NEW_MODEL_NAME = "deepcrust_fleet_v4_cold_pizza"

# 3. How long to retrain (500k is usually enough for adaptation)
TRANSFER_STEPS = 1_000_000


def make_env(rank, seed=0):
    def _init():
        # This loads the CURRENT DeepCrustEnv.py (which is your V5 Balanced version)
        env = DeepCrustEnv()
        env.reset(seed=seed + rank)
        return env

    return _init


def run_transfer_learning():
    num_cpu = 16

    run = wandb.init(
        project="deepcrust-ai",
        config={"setup": "Transfer Learning (V3 -> V5)"},
        sync_tensorboard=True,
    )

    # 1. Create the NEW Environment (V5 Physics)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    print(f"--- LOADING PRETRAINED BRAIN: {PRETRAINED_MODEL} ---")

    # 2. Load the Old Brain into the New Body
    # We pass the new 'env' so the model starts interacting with V5 immediately
    try:
        model = PPO.load(PRETRAINED_MODEL, env=env, device="cuda")
    except FileNotFoundError:
        print(f"❌ Error: Could not find {PRETRAINED_MODEL}.zip")
        return

    # 3. Reset Learning Rate (Optional but recommended)
    # We give it a slightly smaller learning rate so it doesn't "forget" everything
    # instantly, but tweaks its strategy.
    model.learning_rate = 0.0002

    print(f"--- STARTING ADAPTATION (+{TRANSFER_STEPS} Steps) ---")
    print("The model will initially have a weird reward score as it adjusts to new prices.")

    model.learn(
        total_timesteps=TRANSFER_STEPS,
        reset_num_timesteps=True,  # Reset the step counter for this new run
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )

    model.save(NEW_MODEL_NAME)
    print(f"✅ Adaptation Complete! Saved as {NEW_MODEL_NAME}")
    run.finish()


if __name__ == "__main__":
    run_transfer_learning()