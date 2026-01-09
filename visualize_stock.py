import matplotlib

# Try 'Qt5Agg' if installed, otherwise 'TkAgg'
try:
    import PyQt5

    matplotlib.use('Qt5Agg')
except ImportError:
    matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env.DeepCrustEnv import DeepCrustEnv

# --- CONFIGURATION ---
MODEL_V3_PATH = "deepcrust_fleet_v4_cold_pizza"
MODEL_PRO_PATH = "deepcrust_pro_large"
STATS_PATH = "deepcrust_pro_large_vecnormalize.pkl"
ANGER_THRESHOLD = 5.0


# --- MAP RENDERER (Unchanged) ---
def render_map(env, ax, title):
    ax.clear()

    node_colors = ['#2c3e50']  # Hub
    for i in range(1, 6):
        backlog = env.orders[i - 1]
        if backlog > ANGER_THRESHOLD:
            node_colors.append('#e74c3c')  # Red
        elif backlog > 2:
            node_colors.append('#e67e22')  # Orange
        else:
            node_colors.append('#3498db')  # Blue

    nx.draw_networkx_nodes(env.G, pos=env.node_pos, ax=ax, node_color=node_colors, node_size=500)
    nx.draw_networkx_edges(env.G, pos=env.node_pos, ax=ax, edge_color='#bdc3c7', width=2.0, alpha=0.3)

    for i in range(6):
        loc = env.coords[i]
        ax.text(loc[0], loc[1] + 0.15, env.node_names[i], fontsize=8, fontweight='bold', ha='center', zorder=10)
        if i > 0:
            count = int(env.orders[i - 1])
            col = 'red' if count > ANGER_THRESHOLD else 'white'
            ax.text(loc[0], loc[1] - 0.15, f"Orders: {count}", color=col, fontsize=8, ha='center', fontweight='bold')

    for i in range(env.n_trucks):
        color = '#2ecc71' if env.truck_loads[i] > 0 else '#95a5a6'
        if env.truck_timers[i] > 0:
            start = env.coords[env.truck_prev_locs[i]]
            end = env.coords[env.truck_locs[i]]
            progress = 1.0 - (env.truck_timers[i] / env.truck_max_timers[i])
            cx = start[0] + (end[0] - start[0]) * progress
            cy = start[1] + (end[1] - start[1]) * progress
            ax.plot(cx, cy, 'o', color=color, markersize=10, markeredgecolor='white')
        else:
            loc = env.coords[env.truck_locs[i]]
            ax.plot(loc[0] + (i * 0.05), loc[1], 'o', color=color, markersize=10, markeredgecolor='white')

    ax.set_title(title, fontsize=12, fontweight='bold', color='white')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.0, 1.2)
    ax.axis('off')


# --- COMBINED CHART RENDERER ---
def render_combined_chart(ax, history_v3, history_pro):
    ax.clear()

    steps = np.arange(len(history_v3))

    # Plot V3 (Grey)
    ax.plot(steps, history_v3, color='#95a5a6', linewidth=2, label=f"V3 (Blind): ${history_v3[-1]:.0f}")

    # Plot Pro (Green)
    ax.plot(steps, history_pro, color='#2ecc71', linewidth=2, label=f"Pro (Visible): ${history_pro[-1]:.0f}")

    # Highlight the gap
    ax.fill_between(steps, history_v3, history_pro, where=(np.array(history_pro) > np.array(history_v3)),
                    color='#2ecc71', alpha=0.1, interpolate=True)

    # Styling
    ax.set_title("Live Revenue Comparison", fontsize=10, fontweight='bold', color='white')
    ax.grid(color='#444444', linestyle=':', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=8, facecolor='#222222', edgecolor='white')

    # Scrolling Window (Show last 100 steps)
    if len(history_v3) > 100:
        ax.set_xlim(len(history_v3) - 100, len(history_v3))
    else:
        ax.set_xlim(0, 100)

    # Dynamic Y-Axis
    max_val = max(max(history_v3), max(history_pro)) if history_v3 else 0
    ax.set_ylim(min(min(history_v3), min(history_pro)) - 10, max_val + 50)


def adapt_obs_v3(obs):
    return np.delete(obs, [3, 7, 11])


def run_battle():
    print("--- BATTLE ARENA V3 (Combined Chart) ---")

    env_v3 = DeepCrustEnv()

    dummy_env = DummyVecEnv([lambda: DeepCrustEnv()])
    try:
        env_pro_wrapper = VecNormalize.load(STATS_PATH, dummy_env)
        env_pro_wrapper.training = False
        env_pro_wrapper.norm_reward = False
    except:
        print("❌ Error loading stats.")
        return

    model_v3 = PPO.load(MODEL_V3_PATH)
    model_pro = PPO.load(MODEL_PRO_PATH)

    SEED = 42
    obs_v3, _ = env_v3.reset(seed=SEED)
    real_env_pro = env_pro_wrapper.venv.envs[0]
    real_env_pro.reset(seed=SEED)
    obs_pro = env_pro_wrapper.reset()

    revenue_v3_hist = [0]
    revenue_pro_hist = [0]
    total_rev_v3 = 0
    total_rev_pro = 0

    # PLOT SETUP
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax_map_v3 = fig.add_subplot(gs[0, 0])
    ax_map_pro = fig.add_subplot(gs[0, 1])
    # Span the chart across both columns [1, :]
    ax_chart = fig.add_subplot(gs[1, :])

    plt.ion()

    for t in range(500):
        # Actions
        v3_input = adapt_obs_v3(obs_v3)
        action_v3, _ = model_v3.predict(v3_input, deterministic=True)
        action_pro, _ = model_pro.predict(obs_pro, deterministic=True)

        # Steps
        obs_v3, reward_v3, _, _, _ = env_v3.step(action_v3)
        obs_pro, reward_pro_vec, _, _ = env_pro_wrapper.step(action_pro)

        # Stats
        total_rev_v3 += reward_v3
        total_rev_pro += reward_pro_vec[0]
        revenue_v3_hist.append(total_rev_v3)
        revenue_pro_hist.append(total_rev_pro)

        # Render
        if t % 2 == 0:  # Refresh every 2 steps to go faster
            render_map(env_v3, ax_map_v3, "V3 (Blind)")
            render_map(real_env_pro, ax_map_pro, "Pro (Visible)")
            render_combined_chart(ax_chart, revenue_v3_hist, revenue_pro_hist)

            plt.pause(0.01)

        if not plt.fignum_exists(fig.number):
            break

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    run_battle()