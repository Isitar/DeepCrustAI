import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class DeepCrustEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(self, render_mode=None):
        super(DeepCrustEnv, self).__init__()

        # --- CONFIGURATION ---
        self.n_cities = 5
        self.n_trucks = 3
        self.max_capacity = 10
        self.travel_cost = 0.1
        self.delivery_reward = 10
        self.anger_penalty_factor = 0.1
        self.refill_reward = 0.1
        self.render_mode = render_mode

        # --- GRAPH LAYOUT ---
        self.G = nx.star_graph(self.n_cities)
        self.pos = nx.spring_layout(self.G, seed=42)
        self.node_names = {0: "Hub", 1: "Uni", 2: "Burbs", 3: "DownT", 4: "Stad", 5: "Indus"}

        # --- ACTION SPACE (Multi-Discrete) ---
        # Each truck needs its own action (0-5)
        # Result: [Action_Truck_1, Action_Truck_2, Action_Truck_3]
        self.action_space = spaces.MultiDiscrete([self.n_cities + 1] * self.n_trucks)

        # --- OBSERVATION SPACE ---
        # [T1_Loc, T1_Load, T2_Loc, T2_Load, ... , City1_Ord, ... City5_Ord, Time]
        # Size = (2 * n_trucks) + n_cities + 1

        # 1. Truck Limits
        truck_low = [0, 0] * self.n_trucks
        truck_high = [self.n_cities, self.max_capacity] * self.n_trucks

        # 2. City Limits
        city_low = [0] * self.n_cities
        city_high = [200] * self.n_cities

        # Combine
        low = np.array(truck_low + city_low + [0])
        high = np.array(truck_high + city_high + [1000])

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        self.fig, self.ax = None, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Init Trucks: [Loc=0, Load=0] for all trucks
        truck_states = np.zeros(self.n_trucks * 2, dtype=np.float32)

        # Init Orders
        orders = np.zeros(self.n_cities, dtype=np.float32)

        self.time_step = 0
        self.state = np.concatenate((truck_states, orders, [self.time_step])).astype(np.float32)

        return self.state, {}

    def get_demand(self, t, node_idx):
        """Generates 'Rush Hour' demand based on sine waves."""
        lambda_rate = 0.6  # Increased slightly for 3 trucks

        if node_idx == 0:  # Uni
            seasonality = np.sin((t - 80) / 20) + 1.2
        elif node_idx == 2:  # Downtown
            seasonality = np.sin((t - 20) / 20) + 1.2
        else:
            seasonality = 1.0

        if np.random.rand() < (lambda_rate * seasonality * 0.3):
            return 1
        return 0

    def step(self, actions):
        self.time_step += 1

        # -- Parse State --
        # Truck data is the first 2*N elements
        truck_data = self.state[:self.n_trucks * 2]
        orders = self.state[self.n_trucks * 2: -1]

        reward = 0

        # -- PROCESS EACH TRUCK --
        # We loop through trucks and apply their specific action
        for i in range(self.n_trucks):
            # Extract current truck info
            idx_loc = i * 2
            idx_load = i * 2 + 1

            curr_loc = int(truck_data[idx_loc])
            curr_load = int(truck_data[idx_load])

            target_loc = actions[i]  # Action for this specific truck

            # 1. Movement
            if target_loc != curr_loc:
                reward -= self.travel_cost
                curr_loc = target_loc

            # 2. Load/Unload
            if curr_loc == 0:
                # Refill
                if curr_load < self.max_capacity:
                    reward += self.refill_reward  # reward refilling to encourage going back home
                curr_load = self.max_capacity
            else:
                # Deliver
                city_idx = curr_loc - 1
                demand = orders[city_idx]

                # Logic: Multiple trucks might compete for demand in same step
                # The first one in the loop takes it (simple priority)
                delivery = min(curr_load, demand)
                curr_load -= delivery
                orders[city_idx] -= delivery
                reward += (delivery * self.delivery_reward)

            # Write back to local array
            truck_data[idx_loc] = curr_loc
            truck_data[idx_load] = curr_load

        # -- DYNAMICS --
        for i in range(self.n_cities):
            orders[i] += self.get_demand(self.time_step, i)

        # -- PENALTY --
        reward -= (np.sum(orders) * self.anger_penalty_factor)

        # Update Global State
        self.state = np.concatenate((truck_data, orders, [self.time_step])).astype(np.float32)

        terminated = self.time_step >= 200

        if self.render_mode == "human":
            self.render()

        return self.state, reward, terminated, False, {}

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 6), facecolor='#1e1e1e')
            self.ax.set_facecolor('#1e1e1e')
            self.ax.axis('off')

        self.ax.clear()

        truck_data = self.state[:self.n_trucks * 2]
        orders = self.state[self.n_trucks * 2: -1]

        # Draw Network
        nx.draw_networkx_edges(self.G, self.pos, ax=self.ax, edge_color='#555555', width=2)

        node_colors, node_sizes = ['#3498db'], [800]
        for qty in orders:
            if qty < 3:
                col = '#2ecc71'
            elif qty < 8:
                col = '#f1c40f'
            else:
                col = '#e74c3c'
            node_colors.append(col)
            node_sizes.append(600 + (qty * 50))

        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax, node_color=node_colors, node_size=node_sizes)

        labels = {0: "HUB"}
        for i, qty in enumerate(orders):
            labels[i + 1] = f"{self.node_names[i + 1]}\n{int(qty)}"
        nx.draw_networkx_labels(self.G, self.pos, labels=labels, ax=self.ax, font_color='white', font_size=9,
                                font_weight='bold')

        # Draw TRUCKS (Iterate)
        for i in range(self.n_trucks):
            loc = int(truck_data[i * 2])
            load = int(truck_data[i * 2 + 1])

            if loc in self.pos:
                x, y = self.pos[loc]
                # Add Jitter so trucks don't overlap perfectly
                jitter_x = (np.random.rand() - 0.5) * 0.15
                jitter_y = (np.random.rand() - 0.5) * 0.15

                # Different color if empty?
                t_col = 'white' if load > 0 else '#95a5a6'

                self.ax.plot(x + jitter_x, y + jitter_y, 'o', color=t_col, markeredgecolor='blue', markersize=10,
                             markeredgewidth=2)

        self.ax.set_title(f"DeepCrust Fleet - Time: {int(self.state[-1])}", color='white', fontsize=14)
        plt.pause(0.05)