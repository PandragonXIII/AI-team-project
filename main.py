import gymnasium as gym
import Agent
import random
import numpy as np

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

#WEATHERS = {"sun":0, "cloudy":1, "rain":2}
WEATHERS = ["sun", "cloudy", "rain"]
WEATHER_TRANSITION = np.array([[0.7, 0.2, 0.1],
                               [0.15,0.4, 0.45],
                               [0.3, 0.4, 0.3]]) # 随便设的概率

PASSENGER_LOC_PROB = np.array([[0.2, 0.1, 0.1, 0.6],
                               [0.1, 0.4, 0.4, 0.1],
                               [0.7, 0.1, 0.1, 0.1]]) # 随便设的概率

'''def main(AGENT_TYPE = "reinforcement",
        test_times = 500,
        display_times = 10,
        FROG_OF_WAR = False,
        train_times = 3000,
        l_rate = 0.1,
        d_factor = 0.99,
        expl = 1
        ):'''
def main():
    
    #single_test("search",withWeather=True)
    #print("Negative scores",len([a for a in single_test("reinforcement") if a<0]),"/ 500")
    scores = single_test("markov_search", FROG_OF_WAR = True, withWeather = True, test_times=3000)
    scores = np.array(scores)
    print("average score:", np.mean(scores))
    print("standard deviation:", np.std(scores))


    '''env = gym.make("Taxi-v3", render_mode="human")
    # display
    print("-----display-----")
    for _ in range(display_times):
        observation, info = env.reset()
        observation = list(env.decode(observation))
        terminated, truncated = False, False
        if FROG_OF_WAR:
            observation = Agent.AddFrogToObs(env, observation)  
        rewards = 0
        agent.setup()
        for s in range(30):
            action = agent.get_best_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            new_observation = list(env.decode(new_observation))
            if FROG_OF_WAR:
                observation = Agent.AddFrogToObs(env, observation)  
            rewards += reward
            # env.render()
            observation = new_observation

            if terminated or truncated:
                break

    env.close()'''
    

def single_test(AGENT_TYPE = "reinforcement",
        agent = None,
        mute = False,
        train_times = 5000,
        test_times = 10,
        withWeather = False,
        FROG_OF_WAR = False,
        l_rate = 0.1,
        d_factor = 0.99,
        expl = 1,
        visual = False
        )->list:
    """
    single trin-test loop
    return: list of scores in testing, len = test_times
    """
    if visual:
        env = gym.make("Taxi-v3", render_mode="human")
    else:
        env = gym.make("Taxi-v3")

    if not mute:
        print("Agent Type: ",AGENT_TYPE)

    if agent is None:
        if AGENT_TYPE == "random":
            agent = Agent.RandomAgent(env)
            train_times = 0 # random agent does not need training
        elif AGENT_TYPE == "reinforcement":
            agent = Agent.ReinforcementAgent(env,
                            learning_rate=l_rate, discount_factor=d_factor, explore=expl)
        elif AGENT_TYPE == "search":
            agent = Agent.SearchAgent(env) # search agent does not need training
            train_times = 0
        elif AGENT_TYPE == "markov_search":
            agent = Agent.MarkovSearchAgent(env, WEATHER_TRANSITION, PASSENGER_LOC_PROB)
            train_times = 0
        else:
            raise Exception("unknown agent type")

    # start training
    if not mute:
        print("-----training-----")
    for _ in range(train_times):
        # print(".", end="")
        observation, info = env.reset()
        observation = list(env.decode(observation))
        """
        ATTENTION!
        observation = [taxi_row, taxi_col, passenger_location, destination]
        observation is always list now, 除非是在函数中作为index使用
        """
        terminated, truncated = False, False
        if FROG_OF_WAR:
            observation = Agent.AddFrogToObs(env, observation)        
        agent.setup()
        while not( terminated or truncated):
            action = agent.explore(observation)
            old_observation = observation
            observation, reward, terminated, truncated, info = env.step(action)
            observation = list(env.decode(observation))
            if FROG_OF_WAR:
                observation = Agent.AddFrogToObs(env, observation)  
            agent.update(old_observation, action, observation, reward)
    if not mute:
        print("\n")

    # start testing
    scores = []
    weather=random.choice(range(3)) # initial weather: all equally likely
    for _ in range(test_times):
        if withWeather:
            #testcases = []
            #for _ in range(test_times):
            taxi_row = random.choice(range(5))
            taxi_col = random.choice(range(5))
            dest = random.choice(range(4))
            #weather=random.choice(range(3)) # initial weather: all equally likely
            # get pass_loc by probabilities
            r=random.random()
            pass_loc=0
            s=0.0
            for prob in PASSENGER_LOC_PROB[weather]: # get next weather by probabilities
                s+=prob
                if r<=s:
                    break
                pass_loc+=1
                if pass_loc>3: pass_loc=3 # in case potential overflow (chance extremely small)
                #testcases.append(env.encode(taxi_row, taxi_col, pass_loc, dest))
            observation, info = env.reset(state = env.encode(taxi_row, taxi_col, pass_loc, dest))
        else:
            observation, info = env.reset()
        observation = list(env.decode(observation))
        terminated, truncated = False, False
        if not mute:
            print("-----test:{}-----".format(_))
            print(observation)
            if withWeather: 
                print("WEATHER:", WEATHERS[weather])
                print("passenger loc:",env.locs[observation[2]])
                print("taxiloc:", observation[0], observation[1])
        if FROG_OF_WAR:
            observation = Agent.AddFrogToObs(env, observation)  
        total_reward = 0
        agent.setup()
        while not( terminated or truncated):
            action = agent.get_best_action(observation)
            #print("Taking action:",action)
            observation, reward, terminated, truncated, info = env.step(action)
            observation = list(env.decode(observation))
            if FROG_OF_WAR:
                observation = Agent.AddFrogToObs(env, observation)  
            total_reward += reward
        # game will terminate automatically after 200 steps
        # print the final score
        if not mute:
            print("score: ", total_reward)
        scores.append(total_reward)

        if withWeather:
            #change weather
            r=random.random()
            nextweather=0
            s=0.0
            for prob in WEATHER_TRANSITION[weather]: # get next weather by probabilities
                s+=prob
                if r<=s:
                    break
                nextweather+=1
            #if nextweather>2: nextweather=2 # in case potential overflow (chance extremely small)
            weather=nextweather

    env.close()
    return scores
    
if __name__ == "__main__":
    main()