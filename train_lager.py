import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import wandb
from wandb.integration.sb3 import WandbCallback
from env.DeepCrustEnv import DeepCrustEnv
import torch.nn as nn

# --- CONFIGURATION ---
RUN_NAME = "deepcrust_pro_large"
TOTAL_TIMESTEPS = 2_000_000  # Give it time to learn the complex logic


def make_env(rank, seed=0):
    def _init():
        env = DeepCrustEnv()
        env.reset(seed=seed + rank)
        return env

    return _init


def train_pro():
    num_cpu = 16  # Use all cores

    run = wandb.init(
        project="deepcrust-ai",
        config={
            "policy_type": "MlpPolicy",
            "total_timesteps": TOTAL_TIMESTEPS,
            "env_name": "DeepCrust-V5-Visible",
            "architecture": "[256, 256]",  # The Big Brain
            "learning_rate": 0.0003,
        },
        sync_tensorboard=True,
    )

    # 1. Create Env
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 2. Normalize Observations (Crucial for "Age" vs "Location" math)
    # This automatically scales inputs so the NN learns faster
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # 3. Define The "Big Brain" Policy
    # net_arch=[dict(pi=[256, 256], vf=[256, 256])] means:
    # - Actor (Policy) gets 2 layers of 256 neurons
    # - Critic (Value) gets 2 layers of 256 neurons
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=dict(pi=[256, 256], vf=[256, 256])
    )

    print(f"--- STARTING PROFESSIONAL TRAINING: {RUN_NAME} ---")
    print("Architecture: [256, 256] (Standard is [64, 64])")

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,  # Standard robust LR
        n_steps=2048,  # Longer rollout buffer
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,  # Slight exploration bonus
        verbose=1,
        tensorboard_log=f"runs/{RUN_NAME}",
        device="cuda"
    )

    # 4. Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            WandbCallback(gradient_save_freq=100, verbose=2),
            CheckpointCallback(save_freq=100000, save_path=f"./models/{RUN_NAME}", name_prefix="ckpt")
        ]
    )

    # 5. Save Model AND Normalization Stats
    model.save(RUN_NAME)
    env.save(f"{RUN_NAME}_vecnormalize.pkl")  # MUST save this to load correctly later

    print(f"✅ Training Complete. Saved model to {RUN_NAME}")
    print(f"✅ Saved normalization stats to {RUN_NAME}_vecnormalize.pkl")
    run.finish()


if __name__ == "__main__":
    train_pro()