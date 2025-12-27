import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv  # <--- The Multiprocessing Wrapper
from stable_baselines3.common.utils import set_random_seed
import wandb
from wandb.integration.sb3 import WandbCallback
import os

from env.DeepCrustEnv import DeepCrustEnv


# Helper function to create environment instances
def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param rank: index of the subprocess
    :param seed: the initial seed for RNG
    """

    def _init():
        env = DeepCrustEnv()
        env.reset(seed=seed + rank)  # Important: Different seed for each worker!
        return env

    return _init


def train():
    # --- CONFIGURATION ---
    num_cpu = 16
    total_timesteps = 1500000
    steps_per_env = 1024

    run = wandb.init(
        project="deepcrust-ai",
        config={"policy": "MlpPolicy", "n_envs": num_cpu},
        sync_tensorboard=True,
    )

    # --- PARALLELIZATION ---
    # Create 'num_cpu' separate environments
    print(f"--- Spawning {num_cpu} parallel environments ---")
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # Initialize Agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=f"runs/{run.id}",
        learning_rate=0.0003,
        n_steps=steps_per_env,  # Steps per environment before update
        batch_size=256,
        n_epochs=10,
        ent_coef=0.05,
        device="cuda",
    )

    print("--- STARTING PARALLEL TRAINING ---")
    model.learn(
        total_timesteps=total_timesteps,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
    )

    print("--- TRAINING COMPLETE ---")
    model.save("deepcrust_fleet_v3_cold_pizza")
    run.finish()


if __name__ == "__main__":
    train()