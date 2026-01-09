import matplotlib

# Force Headless Mode (Fixes window resizing bugs and allows recording on servers)
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.DeepCrustEnv import DeepCrustEnv

# --- CONFIGURATION ---
# Filenames
MODEL_V4_PATH = "deepcrust_fleet_v4_cold_pizza"  # Or "deepcrust_fleet_v4_cold_pizza"
MODEL_PRO_PATH = "deepcrust_pro_large"
STATS_PATH = "deepcrust_pro_large_vecnormalize.pkl"

OUTPUT_FILE = "battle_royale_slow.gif"
STEPS = 400  # How long the battle lasts
FPS = 15


# --- RENDERERS ---
def render_map(env, ax, title):
    ax.clear()

    # 1. Colors (Anger Logic)
    ANGER_THRESHOLD = 5
    node_colors = ['#2c3e50']  # Hub
    for i in range(1, 6):
        backlog = env.orders[i - 1]
        if backlog > ANGER_THRESHOLD:
            node_colors.append('#e74c3c')  # Red
        elif backlog > 2:
            node_colors.append('#e67e22')  # Orange
        else:
            node_colors.append('#3498db')  # Blue

    # 2. Draw Nodes/Edges
    nx.draw_networkx_nodes(env.G, pos=env.node_pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(env.G, pos=env.node_pos, ax=ax, edge_color='#bdc3c7', width=2.0, alpha=0.3)

    # 3. Text Labels
    for i in range(6):
        loc = env.coords[i]
        ax.text(loc[0], loc[1] + 0.15, env.node_names[i], fontsize=8, fontweight='bold', ha='center', zorder=10)
        if i > 0:
            count = int(env.orders[i - 1])
            col = 'red' if count > ANGER_THRESHOLD else 'white'
            ax.text(loc[0], loc[1] - 0.15, f"Orders: {count}", color=col, fontsize=8, ha='center', fontwe ight='bold')

    # 4. Trucks
    TRUCK_COLORS = ['#00e5ff', '#ffeb3b', '#ff4081']  # Cyan, Yellow, Pink
    for i in range(env.n_trucks):
        # Use specific color for each truck ID
        t_color = TRUCK_COLORS[i % len(TRUCK_COLORS)]

        # If empty, dim the color slightly (optional, or just keep distinct)
        if env.truck_loads[i] == 0:
            t_color = '#7f8c8d'  # Grey if empty

        # Interpolation for smooth movement
        if env.truck_timers[i] > 0:
            start = env.coords[env.truck_prev_locs[i]]
            end = env.coords[env.truck_locs[i]]
            progress = 1.0 - (env.truck_timers[i] / env.truck_max_timers[i])
            cx = start[0] + (end[0] - start[0]) * progress
            cy = start[1] + (end[1] - start[1]) * progress

            ax.plot(cx, cy, 'o', color=t_color, markersize=14, markeredgecolor='white', zorder=20)
            # Show Capacity inside the dot or above it
            ax.text(cx, cy, str(int(env.truck_loads[i])), color='black', fontsize=7, fontweight='bold', ha='center',
                    va='center', zorder=21)
        else:
            loc = env.coords[env.truck_locs[i]]
            # Add jitter so stacked trucks are visible
            jit_x = loc[0] + (i * 0.05)
            ax.plot(jit_x, loc[1], 'o', color=t_color, markersize=14, markeredgecolor='white', zorder=20)
            ax.text(jit_x, loc[1], str(int(env.truck_loads[i])), color='black', fontsize=7, fontweight='bold',
                    ha='center', va='center', zorder=21)

    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.2)
    ax.axis('off')


def render_combined_chart(ax, history_v4, history_pro):
    ax.clear()

    steps = np.arange(len(history_v4))

    # Draw Lines
    ax.plot(steps, history_v4, color='#95a5a6', linewidth=2, label=f"Blind: ${history_v4[-1]:.0f}")
    ax.plot(steps, history_pro, color='#2ecc71', linewidth=2, label=f"Pro: ${history_pro[-1]:.0f}")

    # Highlight the Gap (Money Left on Table)
    ax.fill_between(steps, history_v4, history_pro, where=(np.array(history_pro) > np.array(history_v4)),
                    color='#2ecc71', alpha=0.15, interpolate=True)

    # Styling
    ax.set_title("Live Revenue Comparison", fontsize=10, fontweight='bold', color='white')
    ax.grid(color='#444444', linestyle=':', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=8, facecolor='#222222', edgecolor='white')

    # Rolling Window (Last 100 steps)
    if len(history_v4) > 100:
        ax.set_xlim(len(history_v4) - 100, len(history_v4))
    else:
        ax.set_xlim(0, 100)

    # Dynamic Y-Axis
    max_val = max(max(history_v4), max(history_pro)) if history_v4 else 0
    ax.set_ylim(-10, max_val + 50)


def adapt_obs_v4(obs):
    # Remove Age columns for V4
    return np.delete(obs, [3, 7, 11])


# --- MAIN RECORDING LOOP ---
def record_battle():
    print(f"--- RECORDING BATTLE ROYALE ({STEPS} steps @ {FPS} FPS) ---")

    # 1. Setup Envs
    env_v4 = DeepCrustEnv()

    dummy_env = DummyVecEnv([lambda: DeepCrustEnv()])
    try:
        env_pro_wrapper = VecNormalize.load(STATS_PATH, dummy_env)
        env_pro_wrapper.training = False
        env_pro_wrapper.norm_reward = False
    except:
        print(f"❌ Error: Could not load {STATS_PATH}")
        return

    # 2. Load Models
    try:
        model_v4 = PPO.load(MODEL_V4_PATH)
        model_pro = PPO.load(MODEL_PRO_PATH)
    except:
        print("❌ Error loading models.")
        return

    # 3. Sync & Reset
    SEED = 42
    obs_v4, _ = env_v4.reset(seed=SEED)

    real_env_pro = env_pro_wrapper.venv.envs[0]
    real_env_pro.reset(seed=SEED)
    obs_pro = env_pro_wrapper.reset()

    # 4. Initialize Data
    revenue_v4_hist = [0]
    revenue_pro_hist = [0]
    total_rev_v4 = 0
    total_rev_pro = 0
    frames = []

    # 5. Setup Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax_map_v4 = fig.add_subplot(gs[0, 0])
    ax_map_pro = fig.add_subplot(gs[0, 1])
    ax_chart = fig.add_subplot(gs[1, :])

    # 6. Run Simulation
    for t in range(STEPS):
        # Actions
        v4_input = adapt_obs_v4(obs_v4)
        action_v4, _ = model_v4.predict(v4_input, deterministic=True)
        action_pro, _ = model_pro.predict(obs_pro, deterministic=True)

        # Step
        obs_v4, reward_v4, _, _, _ = env_v4.step(action_v4)
        obs_pro, reward_pro_vec, _, _ = env_pro_wrapper.step(action_pro)

        # Stats
        total_rev_v4 += reward_v4
        total_rev_pro += reward_pro_vec[0]
        revenue_v4_hist.append(total_rev_v4)
        revenue_pro_hist.append(total_rev_pro)

        # RENDER FRAME
        render_map(env_v4, ax_map_v4, "Blind")
        render_map(real_env_pro, ax_map_pro, "Pro")
        render_combined_chart(ax_chart, revenue_v4_hist, revenue_pro_hist)

        # CAPTURE FRAME
        fig.canvas.draw()
        image_rgba = np.array(fig.canvas.buffer_rgba())
        image_rgb = image_rgba[:, :, :3]  # Drop Alpha
        frames.append(image_rgb)

        if t % 20 == 0:
            print(f"   ... captured frame {t}/{STEPS}")

    # 7. Save GIF
    print(f"💾 Saving to {OUTPUT_FILE}...")
    imageio.mimsave(OUTPUT_FILE, frames, fps=FPS, loop=0)
    print("✅ Done! Good luck with the presentation!")
    plt.close()


if __name__ == "__main__":
    record_battle()