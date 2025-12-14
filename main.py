from env.DeepCrustEnv import DeepCrustEnv

env = DeepCrustEnv()
obs, _ = env.reset()

print("--- STARTING RANDOM AGENT ---")
for _ in range(20):
    action = env.action_space.sample()

    obs, reward, done, _, _ = env.step(action)

    env.render()
    print(f"Action: Go to Node {action}, Reward: {reward:.1f}")
    print("-" * 30)

    if done:
        break