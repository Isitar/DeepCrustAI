import matplotlib

# Use Agg for recording to GIF, or TkAgg/Qt5Agg for live viewing
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import imageio
from stable_baselines3 import PPO
from env.DeepCrustEnv import DeepCrustEnv

# --- CONFIGURATION ---
# The Blind "Hero"
MODEL_BLIND_PATH = "deepcrust_fleet_v4_cold_pizza"
# The Visible "Lazy" Agent
MODEL_VISIBLE_PATH = "deepcrust_fleet_v5_visible_clock"

OUTPUT_FILE = "laziness_paradox.gif"
STEPS = 200
FPS = 10


# --- ADAPTERS ---
def adapt_obs_blind(obs):
    # V4 is blind: It expects 15 inputs. Env gives 18.
    # We remove the "Pizza Age" columns (indices 3, 7, 11)
    return np.delete(obs, [3, 7, 11])


# --- RENDERERS ---
def render_map(env, ax, title):
    ax.clear()

    # 1. City Colors (Anger)
    node_colors = ['#2c3e50']  # Hub
    for i in range(1, 6):
        backlog = env.orders[i - 1]
        if backlog > 5:
            node_colors.append('#e74c3c')  # Angry Red
        elif backlog > 2:
            node_colors.append('#e67e22')  # Warning Orange
        else:
            node_colors.append('#3498db')  # Happy Blue

    # 2. Draw Map
    nx.draw_networkx_nodes(env.G, pos=env.node_pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(env.G, pos=env.node_pos, ax=ax, edge_color='#bdc3c7', width=2.0, alpha=0.3)

    # 3. Labels
    for i in range(6):
        loc = env.coords[i]
        ax.text(loc[0], loc[1] + 0.15, env.node_names[i], fontsize=8, fontweight='bold', ha='center', zorder=10)
        if i > 0:
            count = int(env.orders[i - 1])
            col = 'red' if count > 5 else 'white'
            ax.text(loc[0], loc[1] - 0.15, f"{count}", color=col, fontsize=8, ha='center', fontweight='bold')

    # 4. Trucks
    # Grey = Lazy/Idle, Green = Working
    for i in range(env.n_trucks):
        is_moving = env.truck_timers[i] > 0
        has_cargo = env.truck_loads[i] > 0

        # Color Logic:
        # If it has cargo, it's working (Green)
        # If it's empty, it's idle (Grey) -> Watch for V5 having lots of GREY trucks!
        color = '#2ecc71' if has_cargo else '#7f8c8d'

        if is_moving:
            start = env.coords[env.truck_prev_locs[i]]
            end = env.coords[env.truck_locs[i]]
            progress = 1.0 - (env.truck_timers[i] / env.truck_max_timers[i])
            cx = start[0] + (end[0] - start[0]) * progress
            cy = start[1] + (end[1] - start[1]) * progress
            ax.plot(cx, cy, 'o', color=color, markersize=12, markeredgecolor='white', zorder=20)
        else:
            loc = env.coords[env.truck_locs[i]]
            jit = (i * 0.05)
            ax.plot(loc[0] + jit, loc[1], 'o', color=color, markersize=12, markeredgecolor='white', zorder=20)

    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
    ax.axis('off')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.2)


def render_comparison_chart(ax, hist_blind, hist_visible):
    ax.clear()
    steps = np.arange(len(hist_blind))

    # Blind Line (Green - The Winner)
    ax.plot(steps, hist_blind, color='#2ecc71', linewidth=2, label=f"Blind (V4): ${hist_blind[-1]:.0f}")

    # Visible Line (Red/Orange - The Loser)
    ax.plot(steps, hist_visible, color='#e74c3c', linewidth=2, label=f"Visible (V5): ${hist_visible[-1]:.0f}")

    # Fill the gap to emphasize the "Cost of Laziness"
    ax.fill_between(steps, hist_blind, hist_visible, where=(np.array(hist_blind) > np.array(hist_visible)),
                    color='#e74c3c', alpha=0.1)

    ax.set_title("Cumulative Profit (The Laziness Gap)", fontsize=10, fontweight='bold', color='white')
    ax.grid(color='#444444', linestyle=':', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=8, facecolor='#222222', edgecolor='white')

    # Dynamic View
    if len(hist_blind) > 100:
        ax.set_xlim(len(hist_blind) - 100, len(hist_blind))
    else:
        ax.set_xlim(0, 100)

    # Y-Axis
    max_val = max(max(hist_blind), max(hist_visible)) if hist_blind else 0
    min_val = min(min(hist_blind), min(hist_visible)) if hist_blind else 0
    ax.set_ylim(min_val - 50, max_val + 50)


# --- MAIN ---
def run_laziness_test():
    print("--- GENERATING LAZINESS PARADOX GIF ---")

    # 1. Setup Environments
    env_blind = DeepCrustEnv()
    env_visible = DeepCrustEnv()

    # 2. Load Models
    try:
        model_blind = PPO.load(MODEL_BLIND_PATH)
        # Assuming V5 was standard (not normalized)
        model_visible = PPO.load(MODEL_VISIBLE_PATH)
        print("✅ Models loaded.")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return

    # 3. Sync
    SEED = 123
    obs_blind, _ = env_blind.reset(seed=SEED)
    obs_visible, _ = env_visible.reset(seed=SEED)

    hist_blind, hist_visible = [0], [0]
    cum_blind, cum_visible = 0, 0
    frames = []

    # 4. Plot Setup
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1])

    ax_map_blind = fig.add_subplot(gs[0, 0])
    ax_map_visible = fig.add_subplot(gs[0, 1])
    ax_chart = fig.add_subplot(gs[1, :])  # Full width chart

    for t in range(STEPS):
        # Actions
        # V4 needs adapter
        act_blind, _ = model_blind.predict(adapt_obs_blind(obs_blind), deterministic=True)
        # V5 sees everything (18 inputs)
        act_visible, _ = model_visible.predict(obs_visible, deterministic=True)

        # Steps
        obs_blind, r_blind, _, _, _ = env_blind.step(act_blind)
        obs_visible, r_visible, _, _, _ = env_visible.step(act_visible)

        cum_blind += r_blind
        cum_visible += r_visible
        hist_blind.append(cum_blind)
        hist_visible.append(cum_visible)

        # Render
        render_map(env_blind, ax_map_blind, "V4 (Blind)\nHappy & Hardworking")
        render_map(env_visible, ax_map_visible, "V5 (Visible)\nAnxious & Lazy")
        render_comparison_chart(ax_chart, hist_blind, hist_visible)

        # Capture
        fig.canvas.draw()
        img = np.array(fig.canvas.buffer_rgba())[:, :, :3]
        frames.append(img)

        if t % 50 == 0: print(f"Frame {t}/{STEPS}")

    # Save
    print(f"💾 Saving {OUTPUT_FILE}...")
    imageio.mimsave(OUTPUT_FILE, frames, fps=FPS, loop=0)
    print("✅ Done!")
    plt.close()


if __name__ == "__main__":
    run_laziness_test()