import gymnasium as gym
from gymnasium import spaces
import numpy as np


class DeepCrustEnv(gym.Env):
    def __init__(self):
        super(DeepCrustEnv, self).__init__()

        # --- CONFIGURATION ---
        self.n_cities = 5
        self.max_capacity = 10
        self.travel_cost = 1
        self.delivery_reward = 20
        self.anger_penalty_factor = 0.5

        # --- ACTION SPACE ---
        # 0 = Go to Hub
        # 1-5 = Go to City 1-5
        self.action_space = spaces.Discrete(self.n_cities + 1)

        # --- OBSERVATION SPACE ---
        # [Truck_Location, Truck_Load, City1_Orders, ..., City5_Orders]
        # Low/High limits for the vector
        low = np.array([0, 0] + [0] * self.n_cities)
        high = np.array([self.n_cities, self.max_capacity] + [100] * self.n_cities)

        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

        self.state = None
        self.time_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initial State: Truck at Hub (0), Empty Load (0), 0 Orders everywhere
        self.state = np.zeros(2 + self.n_cities, dtype=np.int32)
        self.time_step = 0

        return self.state, {}

    def step(self, action):
        self.time_step += 1

        # Unpack State
        truck_loc = self.state[0]
        truck_load = self.state[1]
        orders = self.state[2:]  # Array of 5 cities

        reward = 0

        # --- 1. MOVEMENT & LOGIC ---
        target_loc = action

        # If we moved, pay fuel cost
        if target_loc != truck_loc:
            reward -= self.travel_cost
            truck_loc = target_loc  # Instant teleport for Iteration 1

        # If at Hub (0): REFILL
        if truck_loc == 0:
            truck_load = self.max_capacity

        # If at a City (1-5): DELIVER
        elif truck_loc > 0:
            city_idx = truck_loc - 1  # Map node 1..5 to index 0..4
            demand = orders[city_idx]

            # How much can we give?
            delivery_amount = min(truck_load, demand)

            # Update state
            truck_load -= delivery_amount
            orders[city_idx] -= delivery_amount

            # Get paid!
            reward += (delivery_amount * self.delivery_reward)

        # --- 2. GENERATE NEW DEMAND (Random for now) ---
        # Every step, small chance a city gets an order
        for i in range(self.n_cities):
            if np.random.rand() < 0.3:  # 30% chance per step
                orders[i] += 1

        # --- 3. CALCULATE ANGER (Penalty) ---
        # The sum of all waiting orders hurts the score
        total_backlog = np.sum(orders)
        reward -= (total_backlog * self.anger_penalty_factor)

        # Update self.state
        self.state = np.concatenate(([truck_loc, truck_load], orders)).astype(np.int32)

        # Check termination (End after 100 steps for this test)
        terminated = self.time_step >= 100
        truncated = False  # Standard Gym requirement

        return self.state, reward, terminated, truncated, {}

    def render(self):
        # Text-based render for debugging Iteration 1
        print(f"Step {self.time_step} | Loc: {self.state[0]} | Load: {self.state[1]} | Orders: {self.state[2:]}")
