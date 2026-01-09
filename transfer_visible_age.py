import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback
from env.DeepCrustEnv import DeepCrustEnv
import os

# --- CONFIGURATION ---
OLD_MODEL_PATH = "deepcrust_fleet_v4_cold_pizza"  # Your best "Blind" model
NEW_MODEL_NAME = "deepcrust_fleet_v5_visible_clock"
STEPS = 1_000_000


def make_env(rank, seed=0):
    def _init():
        # This loads the NEW env (with extra observation inputs)
        env = DeepCrustEnv()
        env.reset(seed=seed + rank)
        return env

    return _init


def surgery_transfer():
    num_cpu = 16

    run = wandb.init(
        project="deepcrust-ai",
        config={"setup": "Surgery Transfer (New Obs Space)"},
        sync_tensorboard=True,
    )

    # 1. Create the NEW Environment (Larger Observation Space)
    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])

    # 2. Initialize a FRESH Model (Random Weights) with the new shape
    print("--- CREATING NEW MODEL (V5) ---")
    new_model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./runs/")

    # 3. Load the OLD Model (V4)
    print(f"--- LOADING OLD DONOR MODEL ({OLD_MODEL_PATH}) ---")
    if not os.path.exists(f"{OLD_MODEL_PATH}.zip"):
        print("❌ Error: Old model not found.")
        return

    # We load it without an environment just to get the weights
    old_model = PPO.load(OLD_MODEL_PATH, device="cpu")

    # 4. PERFORM THE SURGERY
    print("--- PERFORMING WEIGHT TRANSPLANT ---")

    # Get the policy networks
    old_policy = old_model.policy
    new_policy = new_model.policy

    # A. Feature Extractor (The "Eyes")
    # The first layer is where the mismatch is.
    # Old shape: [64, Old_Obs] | New shape: [64, New_Obs]
    old_weight = old_policy.mlp_extractor.policy_net[0].weight.data
    old_bias = old_policy.mlp_extractor.policy_net[0].bias.data

    # Check dimensions
    # new_model.policy.mlp_extractor.policy_net[0] is the first Linear layer
    new_layer = new_policy.mlp_extractor.policy_net[0]

    # Copy the matching parts
    # We copy columns 0 to N from the old model to the new model.
    # The new columns (at the end) will stay random/zero.
    n_old_inputs = old_weight.shape[1]

    # Copy Weights
    new_layer.weight.data[:, :n_old_inputs] = old_weight
    # Copy Bias (Bias doesn't depend on input size, so it matches perfectly)
    new_layer.bias.data[:] = old_bias

    print(f"✅ Transplanted Input Layer (Kept {n_old_inputs} weights, initialized rest)")

    # B. The Rest of the Network (The "Brain")
    # Since only the input size changed, the rest of the network (hidden layers -> output)
    # is identical in size. We can direct copy.

    # Copy remaining policy layers (Layer 2, Output, etc.)
    # Note: loop starts at 2 because index 0 is the input layer we just handled
    for i in range(2, len(new_policy.mlp_extractor.policy_net)):
        if isinstance(new_policy.mlp_extractor.policy_net[i], nn.Linear):
            new_policy.mlp_extractor.policy_net[i].load_state_dict(
                old_policy.mlp_extractor.policy_net[i].state_dict()
            )

    # Copy Value Network (Critic) - Same logic applies
    # 1. Input Layer (Mismatch)
    old_v_weight = old_policy.mlp_extractor.value_net[0].weight.data
    new_v_layer = new_policy.mlp_extractor.value_net[0]
    new_v_layer.weight.data[:, :n_old_inputs] = old_v_weight
    new_v_layer.bias.data[:] = old_policy.mlp_extractor.value_net[0].bias.data

    # 2. Hidden Layers (Match)
    for i in range(2, len(new_policy.mlp_extractor.value_net)):
        if isinstance(new_policy.mlp_extractor.value_net[i], nn.Linear):
            new_policy.mlp_extractor.value_net[i].load_state_dict(
                old_policy.mlp_extractor.value_net[i].state_dict()
            )

    # Copy Action Head (Final Output) - Matches perfectly
    new_policy.action_net.load_state_dict(old_policy.action_net.state_dict())
    new_policy.value_net.load_state_dict(old_policy.value_net.state_dict())

    print("✅ Surgery Complete. The model remembers how to drive!")
    print("--- STARTING TRAINING (Learning to use the new 'Freshness' input) ---")

    # 5. Train
    # We use a lower learning rate so we don't shock the brain too hard
    new_model.learning_rate = 0.0002

    new_model.learn(
        total_timesteps=STEPS,
        callback=WandbCallback(gradient_save_freq=100, verbose=2),
    )

    new_model.save(NEW_MODEL_NAME)
    print(f"Saved V5 model to {NEW_MODEL_NAME}")
    run.finish()


if __name__ == "__main__":
    surgery_transfer()