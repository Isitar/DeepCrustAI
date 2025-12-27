from math import ceil

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

        self.truck_speed = 0.2

        self.travel_cost = 0.05
        self.delivery_reward = 10
        self.anger_penalty_factor = 0.2
        self.loitering_penalty = 0.5
        #self.refill_reward = 0.1 # maybe add later to increase training rewards

        self.render_mode = render_mode

        # --- GRAPH LAYOUT ---
        self.node_names = {
            0: "FHNW (Hub)",
            1: "Brugg Bhf",
            2: "Neumarkt",
            3: "Königsfelden",
            4: "Vindonissa",
            5: "Industrie"
        }

        # Coordinates (X, Y) relative to FHNW
        self.coords = np.array([
            [0.0, 0.0],  # 0. FHNW
            [-0.5, 0.2],  # 1. Bahnhof
            [-0.8, 0.3],  # 2. Neumarkt
            [-0.3, 0.8],  # 3. Königsfelden
            [0.6, 0.4],  # 4. Vindonissa
            [0.8, -0.6]  # 5. Industrie
        ])
        self.node_pos = {i: self.coords[i] for i in range(6)}
        self.dist_matrix = np.zeros((6, 6))
        for i in range(6):
            for j in range(6):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                self.dist_matrix[i][j] = dist



        self.action_space = spaces.MultiDiscrete([self.n_cities + 1] * self.n_trucks)


        # --- OBSERVATION SPACE ---
        # [T1_Loc, T1_Load, T1_Timer, T2_Loc, T2_Load, T2_Timer ... , City1_Ord, ... City5_Ord, Time]
        # Size = (3 * n_trucks) + n_cities + time
        obs_dim = (self.n_trucks * 3) + self.n_cities + 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(obs_dim,), dtype=np.float32)

        self.fig, self.ax = None, None
        self.G = nx.Graph()
        for i in range(6): self.G.add_node(i, pos=self.coords[i])

        self.pos = nx.spring_layout(self.G, seed=42)

        self.fig, self.ax = None, None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.truck_locs = np.zeros(self.n_trucks, dtype=np.int32)
        self.truck_loads = np.full(self.n_trucks, self.max_capacity, dtype=np.int32)
        self.truck_timers = np.zeros(self.n_trucks, dtype=np.float32)
        self.orders = np.zeros(self.n_cities, dtype=np.float32)
        self.time_step = 0

        self.truck_prev_locs = np.zeros(self.n_trucks, dtype=np.int32)
        self.truck_max_timers = np.ones(self.n_trucks, dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        obs = []
        for i in range(self.n_trucks):
            obs.append(self.truck_locs[i] / self.n_cities)
            obs.append(self.truck_loads[i] / self.max_capacity)
            obs.append(min(self.truck_timers[i], 20.0) / 20.0)
        for i in range(self.n_cities):
            obs.append(min(self.orders[i], 20) / 20.0)
        obs.append(self.time_step / 200.0)
        return np.array(obs, dtype=np.float32)

    def get_demand(self, t, node_idx):
        """
                Generates demand based on the location type in Brugg.
                t: current time step (0-200)
                node_idx: 1=Bahnhof, 2=Neumarkt, 3=Königsfelden, 4=Vindonissa, 5=Industrie
                """
        base_prob = 0.15  # Base chance (15%)

        # 1. BRUGG BAHNHOF (Commuters)
        # Two spikes: Morning (t=20-50) and Evening (t=160-190)
        if node_idx == 1:
            if (20 < t < 50) or (160 < t < 190):
                return 1 if np.random.rand() < 0.6 else 0  # 60% chance during rush
            return 1 if np.random.rand() < 0.05 else 0  # Quiet otherwise

        # 2. NEUMARKT (Shopping Mall)
        # One big peak in the middle of the day (Lunch/Afternoon)
        elif node_idx == 2:
            seasonality = np.sin((t - 100) / 30) + 1.0  # Smooth wave centered at t=100
            # Cap probability at ~50%
            prob = min(0.5, base_prob * seasonality * 1.5)
            return 1 if np.random.rand() < prob else 0

        # 3. KÖNIGSFELDEN (Hospital/Residential) & 4. VINDONISSA
        # Random, steady demand (People always eat pizza)
        elif node_idx in [3, 4]:
            return 1 if np.random.rand() < 0.15 else 0

        # 5. INDUSTRIE (Factories)
        # Constant, heavy demand (Workers need lunch), but drops off at night
        elif node_idx == 5:
            if t < 150:  # Work shift
                return 1 if np.random.rand() < 0.25 else 0
            else:  # Shift over
                return 1 if np.random.rand() < 0.05 else 0

        return 0
    def step(self, actions):
        self.time_step += 1
        reward = 0
        for i in range(self.n_trucks):
            # 1. HANDLE TRAVEL
            if self.truck_timers[i] > 0:
                self.truck_timers[i] -= 1
                if self.truck_timers[i] <= 0:
                    self.truck_timers[i] = 0
                    if self.truck_locs[i] == 0:
                        self.truck_loads[i] = self.max_capacity # refill when in depot
                continue

            # 2. HANDLE ACTIONS
            target = actions[i]
            curr_loc = self.truck_locs[i]

            if curr_loc != 0 and self.truck_loads[i] == 0:
                reward -= self.loitering_penalty  # GO HOME!

            if target != curr_loc:
                dist = self.dist_matrix[curr_loc][target]
                travel_time = ceil(dist / self.truck_speed)

                self.truck_timers[i] = travel_time
                self.truck_max_timers[i] = travel_time  # Save for animation
                self.truck_prev_locs[i] = curr_loc  # Save Start Node
                self.truck_locs[i] = target

                reward -= (dist * self.travel_cost)
            else:
                if curr_loc > 0:
                    city_idx = curr_loc - 1
                    demand = self.orders[city_idx]
                    if demand > 0 and self.truck_loads[i] > 0:
                        delivery = min(self.truck_loads[i], demand)
                        self.truck_loads[i] -= delivery
                        self.orders[city_idx] -= delivery
                        reward += (delivery * self.delivery_reward)

            # 3. DYNAMICS
        for i in range(self.n_cities):
            self.orders[i] += self.get_demand(self.time_step, i + 1)

        reward -= (np.sum(self.orders) * self.anger_penalty_factor)

        terminated = self.time_step >= 200

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, False, {}

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        self.ax.clear()

        # Draw Nodes (No Auto-Labels)
        nx.draw_networkx_nodes(self.G, pos=self.node_pos, ax=self.ax, node_color='#34495e', node_size=600)
        nx.draw_networkx_edges(self.G, pos=self.node_pos, ax=self.ax, edge_color='#bdc3c7', width=1.5, alpha=0.5)

        # --- FIX: Manual Labels (Black, Offset upwards) ---
        for i in range(6):
            loc = self.coords[i]
            # Offset Y by +0.15 so it floats above the node
            self.ax.text(loc[0], loc[1] + 0.15, self.node_names[i],
                         color='black', fontsize=10, fontweight='bold',
                         ha='center', va='bottom', zorder=10)

        # Draw Trucks
        colors = ['#e74c3c', '#f1c40f', '#2ecc71']
        for i in range(self.n_trucks):
            if self.truck_timers[i] > 0:
                # --- INTERPOLATION LOGIC ---
                # 1. Get Start and End coords
                start_pos = self.coords[self.truck_prev_locs[i]]
                end_pos = self.coords[self.truck_locs[i]]

                # 2. Calculate Progress (0.0 to 1.0)
                # If timer = max, progress = 0. If timer = 0, progress = 1.
                progress = 1.0 - (self.truck_timers[i] / self.truck_max_timers[i])

                # 3. Linear Interpolation
                current_x = start_pos[0] + (end_pos[0] - start_pos[0]) * progress
                current_y = start_pos[1] + (end_pos[1] - start_pos[1]) * progress

                # Draw Truck 'Ghost' on the road
                self.ax.plot(current_x, current_y, 'o', color=colors[i], markersize=12, markeredgecolor='white',
                             alpha=0.8, zorder=20)
                self.ax.text(current_x, current_y + 0.08, f"T{i}", color=colors[i], fontsize=8, fontweight='bold',
                             ha='center')

            else:
                # Stationary Truck
                loc = self.coords[self.truck_locs[i]]

                # Offset slightly so multiple trucks don't overlap perfectly
                offset = (i * 0.03)

                self.ax.plot(loc[0] + offset, loc[1], 'o', color=colors[i], markersize=16, markeredgecolor='white',
                             zorder=20)
                # Show Load inside the dot
                self.ax.text(loc[0] + offset, loc[1], str(int(self.truck_loads[i])), color='white', ha='center',
                             va='center', fontweight='bold', fontsize=9, zorder=21)

        # Draw Demand (Red Text Below)
        for i in range(self.n_cities):
            loc = self.coords[i + 1]
            if self.orders[i] > 0:
                self.ax.text(loc[0], loc[1] - 0.15, f"Orders: {int(self.orders[i])}", color='#c0392b',
                             fontweight='bold', ha='center')

        self.ax.set_title(f"FHNW Logistics (Brugg/Windisch) - Step {self.time_step}")
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        plt.pause(0.01)