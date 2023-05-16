import gymnasium as gym

env = gym.make("Taxi-v3")
'''
reinformance learning verion
'''
observation, info = env.reset()
train_times = 1000
trail_time_limit = 200 # 200 steps

for _ in range(train_times):
    pass


env.close()