import matplotlib

matplotlib.use('Qt5Agg')  # or 'TkAgg' depending on your system.
# If this crashes, try commenting it out or using 'Agg' and saving a GIF instead.

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.DeepCrustEnv import DeepCrustEnv

# --- CONFIGURATION ---
MODEL_V3_PATH = "deepcrust_fleet_v4_cold_pizza"  # The Old Model
MODEL_PRO_PATH = "deepcrust_pro_large"  # The New Model
STATS_PATH = "deepcrust_pro_large_vecnormalize.pkl"
ANGER_THRESHOLD = 5.0  # If orders > 5, city turns RED


# --- CUSTOM RENDERER (The Magic) ---
def render_custom(env, ax, title):
    """
    Draws the environment state onto a specific Matplotlib axis (ax).
    This allows us to have side-by-side plots.
    """
    ax.clear()

    # 1. Dynamic Node Colors (Anger Logic)
    node_colors = []
    # Hub (Node 0) is always distinct
    node_colors.append('#2c3e50')

    for i in range(1, 6):  # Cities 1-5
        # Check backlog in the 'orders' array (shifted by -1 because orders is size 5)
        backlog = env.orders[i - 1]
        if backlog > ANGER_THRESHOLD:
            node_colors.append('#e74c3c')  # ANGRY RED
        elif backlog > 2:
            node_colors.append('#e67e22')  # WARNING ORANGE
        else:
            node_colors.append('#3498db')  # HAPPY BLUE

    # 2. Draw Map
    # We use the env's internal graph G and pos
    nx.draw_networkx_nodes(env.G, pos=env.node_pos, ax=ax, node_color=node_colors, node_size=600)
    nx.draw_networkx_edges(env.G, pos=env.node_pos, ax=ax, edge_color='#bdc3c7', width=2.0, alpha=0.3)

    # 3. Draw Labels
    for i in range(6):
        loc = env.coords[i]
        ax.text(loc[0], loc[1] + 0.15, env.node_names[i], fontsize=8, fontweight='bold', ha='center', zorder=10)

        # Show Order Count
        if i > 0:
            count = int(env.orders[i - 1])
            if count > 0:
                ax.text(loc[0], loc[1] - 0.15, f"Orders: {count}", color='red' if count > ANGER_THRESHOLD else 'black',
                        fontsize=8, ha='center')

    # 4. Draw Trucks
    for i in range(env.n_trucks):
        # Color logic based on state
        if env.truck_loads[i] == 0:
            t_color = '#95a5a6'  # Empty (Grey)
        else:
            t_color = '#2ecc71'  # Full (Green)

        # Movement Interpolation
        if env.truck_timers[i] > 0:
            # Moving
            start = env.coords[env.truck_prev_locs[i]]
            end = env.coords[env.truck_locs[i]]
            progress = 1.0 - (env.truck_timers[i] / env.truck_max_timers[i])

            curr_x = start[0] + (end[0] - start[0]) * progress
            curr_y = start[1] + (end[1] - start[1]) * progress

            ax.plot(curr_x, curr_y, 'o', color=t_color, markersize=10, markeredgecolor='white')
            ax.text(curr_x, curr_y + 0.05, f"T{i}", fontsize=7, ha='center')
        else:
            # Stationary
            loc = env.coords[env.truck_locs[i]]
            offset = (i * 0.05) - 0.07  # Slight jitter so they don't overlap perfectly
            ax.plot(loc[0] + offset, loc[1], 'o', color=t_color, markersize=12, markeredgecolor='white')

    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.2)
    ax.axis('off')  # Hide X/Y axis numbers


# --- ADAPTER ---
def adapt_obs_v3(obs):
    # V3 expects 15 inputs, but env gives 18. Remove Age indices [3, 7, 11]
    return np.delete(obs, [3, 7, 11])


def run_battle():
    print("--- SETTING UP THE BATTLE ARENA ---")

    # 1. Setup V3 Environment (Standard)
    env_v3 = DeepCrustEnv()

    # 2. Setup Pro Environment (Normalized)
    # We must wrap it exactly like training
    dummy_env = DummyVecEnv([lambda: DeepCrustEnv()])
    try:
        env_pro_wrapper = VecNormalize.load(STATS_PATH, dummy_env)
        env_pro_wrapper.training = False
        env_pro_wrapper.norm_reward = False
    except:
        print(f"❌ Error loading {STATS_PATH}")
        return

    # 3. Load Models
    try:
        model_v3 = PPO.load(MODEL_V3_PATH)
        model_pro = PPO.load(MODEL_PRO_PATH)
    except:
        print("❌ Error loading models. Check filenames.")
        return

    # 4. Synchronization (The Magic)
    SEED = 123
    obs_v3, _ = env_v3.reset(seed=SEED)

    # Reset wrapper and get internal env
    # Note: VecNormalize requires us to reset the wrapper to get the initial scaled obs
    obs_pro = env_pro_wrapper.reset()

    # Access the "Real" underlying env for Pro so we can read its unscaled state for rendering
    real_env_pro = env_pro_wrapper.venv.envs[0]
    # We must re-seed the internal env strictly to match V3
    real_env_pro.reset(seed=SEED)

    # **CRITICAL FIX**: After manual re-seeding, the wrapper's 'obs' might be stale.
    # We force a fresh observation from the wrapper.
    obs_pro = env_pro_wrapper.reset()

    # 5. Visualization Loop
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    print("--- BATTLE START ---")

    for t in range(500):
        # --- A. GET ACTIONS ---

        # V3 Action (With Adapter)
        v3_input = adapt_obs_v3(obs_v3)
        action_v3, _ = model_v3.predict(v3_input, deterministic=True)

        # Pro Action (Direct, wrapper handles normalization)
        action_pro, _ = model_pro.predict(obs_pro, deterministic=True)

        # --- B. STEP ENVIRONMENTS ---
        obs_v3, _, _, _, _ = env_v3.step(action_v3)

        # VecEnv step returns (obs, rewards, dones, infos)
        obs_pro, _, _, _ = env_pro_wrapper.step(action_pro)

        # --- C. RENDER BATTLE ---
        # We pass the env_v3 and real_env_pro (the one with the actual data)
        render_custom(env_v3, ax1, f"V3 (Blind) - Step {t}\nBacklog: {int(np.sum(env_v3.orders))}")
        render_custom(real_env_pro, ax2, f"PRO (Visible) - Step {t}\nBacklog: {int(np.sum(real_env_pro.orders))}")

        plt.pause(0.1)  # Adjust speed here

        # Check for window close
        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_battle()