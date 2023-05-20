import gymnasium as gym

env = gym.make("Taxi-v3", render_mode="human")
'''
template random taxi agent
'''
for __ in range(5):
    state=env.encode(4,4,0,1) # taxi row 0-4, taxi col 0-4, passgr loc 0-3, passgr dest 0-3
    observation, info = env.reset(state=state)

    for _ in range(2):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        print(action)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

env.close()