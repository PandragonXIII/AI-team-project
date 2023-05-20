import gymnasium as gym
import Agent

'''
env.action_space: 
    0: Move south (down)
    1: Move north (up)
    2: Move east (right)
    3: Move west (left)
    4: Pickup passenger
    5: Drop off passenger

env.observation_space:
(int)((taxi_row * 5 + taxi_col) * 5 + passenger_location) * 4 
    + destination (神奇的编码方式, 取模来解码)
    taxi_row
    taxi_col
    passenger_location: 0: R, 1: G, 2: Y, 3: B, 4: in taxi
    destination: 0: R, 1: G, 2: Y, 3: B

Rewards:
    -1 per step unless other reward is triggered.
    +20 delivering passenger.
    -10 executing “pickup” and “drop-off” actions illegally.

'''
env = gym.make("Taxi-v3")
AGENT_TYPE = "reinforcement"
test_times = 0

"""
note that we only have one passenger in each episode for now.
车的行为是确定性的, 需要学习的应该是墙的位置和乘客的位置
"""
if AGENT_TYPE == "random":
    agent = Agent.RandomAgent(env.observation_space, env.action_space)
    train_times = 0 # random agent does not need training
elif AGENT_TYPE == "reinforcement":
    agent = Agent.ReinforcementAgent(env.observation_space, env.action_space,
                                     learning_rate=0.05, discount_factor=0.99,
                                     explore=1)
    train_times = 5000
else:
    raise Exception("unknown agent type")

# start training
print("-----training-----")
for _ in range(train_times):
    print(".", end="")
    observation, info = env.reset()
    terminated, truncated = False, False
    while not( terminated or truncated):
        action = agent.explore(observation)
        old_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)
        agent.update(old_observation, action, observation, reward)
print("\n")

# start testing
for _ in range(test_times):
    observation, info = env.reset()
    terminated, truncated = False, False
    print("-----test:{}-----".format(_))
    total_reward = 0
    while not( terminated or truncated):
        action = agent.get_best_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
    # game will terminate automatically after 200 steps
    # print the final score
    print("score: ", total_reward)
env.close()

env = gym.make("Taxi-v3", render_mode="human")
# display
print("-----display-----")
for _ in range(10):
    observation, info = env.reset()
    terminated, truncated = False, False
    rewards = 0
    for s in range(30):
        action = agent.get_best_action(observation)
        new_observation, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        # env.render()
        observation = new_observation

        if terminated or truncated:
            break

env.close()