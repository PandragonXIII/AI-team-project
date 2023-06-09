"""
visualize the profermance of the model under different parameters
"""
import matplotlib.pyplot as plt
import numpy as np
import main
import gymnasium as gym
import Agent


def ReLU(x):
    return np.maximum(0, x)

def show_performance():
    AGENT_TYPE = "reinforcement"
    test_times = 100
    FROG_OF_WAR = False
    train_time_range = [1200,15000,600]
    # train_time_range = [1200, 4000, 300]
    l_rate = [0.05, 1, 0.05]
    d_factor = 0.99
    expl = 1

    env = gym.make("Taxi-v3")
    

    x_data = np.arange(train_time_range[0], train_time_range[1], train_time_range[2])
    x_data = np.append([0], x_data)
    y_data = np.arange(l_rate[0], l_rate[1], l_rate[2])
    y_data = np.around(y_data, decimals=2)# 格式化y_data到两位小数
    z_data = []
    for learn_rate in y_data:
        agent = main.Agent.ReinforcementAgent(env=env,
                    learning_rate=learn_rate, discount_factor=d_factor, explore=expl)
        for idx in range(len(x_data)-1):
            train_times = x_data[idx+1]-x_data[idx]
            # train_times = x_data[idx+1]
            print("train times: ", x_data[idx+1])
            scores = main.single_test(AGENT_TYPE,agent,True,train_times,test_times)
            avg_score = np.mean(scores)
            z_data.append(avg_score)
            avg_score = ReLU(avg_score+10)
            print("score: ", avg_score)
    # 绘制热力图
    x_data = x_data[1:]
    z_data = np.array(z_data).reshape(len(y_data), len(x_data))
    plt.imshow(z_data, cmap=plt.cm.hot, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(x_data)), x_data)
    plt.yticks(np.arange(len(y_data)), y_data)
    plt.xlabel("train times")
    plt.ylabel("learning rate")
    plt.title("train times - learning rate - average score")
    # plt.show()

    # note the score of search agent
    # agent = main.Agent.SearchAgent(env=env)
    # train_times = 0
    # scores = main.single_test('search',None,True,train_times,test_times,
    #                             False ,FROG_OF_WAR,l_rate,d_factor,expl)
    # avg_score = ReLU(np.mean(scores))
    # print("search agent score: ", avg_score)
    plt.show()

    #store z_data in file
    np.savetxt("z_data.txt", z_data, fmt="%f", delimiter=",")

    return

# show_performance()

def single_rein_tests():
    AGENT_TYPE = "reinforcement"
    test_times = 50
    FROG_OF_WAR = False
    train_times = 7000
    l_rate = 0.1
    d_factor = 0.99
    expl = 1

    env = gym.make("Taxi-v3")
    agent = main.Agent.ReinforcementAgent(env=env,
                    learning_rate=l_rate, discount_factor=d_factor, explore=expl)
    scores = main.single_test(AGENT_TYPE,agent,True,train_times,test_times)
    avg_score = ReLU(np.mean(scores))
    print("scores: ", scores)
    print("average score: ", avg_score)
    return

# single_rein_tests()

def comp_different_agents():
    test_times = 1000
    FROG_OF_WAR = False
    train_times = [1650,2000,0]
    l_rate = 0.6
    d_factor = 0.99
    expl = 1
    x_data = ["rein-1650","rein-2000", "search"]

    agent_type = ["reinforcement", "reinforcement", "search"]
    y_data = []
    for idx in range(len(x_data)):
        scores = main.single_test(agent_type[idx],None,True,train_times[idx],test_times,
                                  False ,FROG_OF_WAR,l_rate,d_factor,expl)
        y_data.append(scores)
    # print(y_data)
    # plot barplot and error bar
    y_data = np.array(y_data)
    y_mean = np.mean(y_data, axis=1)
    y_error = np.std(y_data, axis=1)
    plt.bar(x_data, y_mean, yerr=y_error, capsize=10)
    plt.xlabel("agent type")
    plt.ylabel("average score over %d times" % test_times)
    plt.title("agent type - average score")
    plt.show()

    # plt.bar(x_data, y_data)
    # plt.xlabel("agent type")
    # plt.ylabel("average score")
    # plt.title("agent type - average score")
    # plt.show()

    return

# comp_different_agents()
def single_visualize():
    AGENT_TYPE = "reinforcement"
    test_times = 100
    FROG_OF_WAR = False
    train_times = 6000
    l_rate = 0.1
    d_factor = 0.99
    expl = 1
    display_times = 5

    env = gym.make("Taxi-v3")
    agent = main.Agent.ReinforcementAgent(env=env,
                    learning_rate=l_rate, discount_factor=d_factor, explore=expl)
    scores = main.single_test(AGENT_TYPE,agent,True,train_times,test_times)
    avg_score = np.mean(scores)
    print("scores: ", scores)
    print("average score: ", avg_score)


    env = gym.make("Taxi-v3", render_mode="human")
    # display
    print("-----display-----")
    for _ in range(display_times):
        observation, info = env.reset()
        observation = list(env.decode(observation))
        terminated, truncated = False, False
        if FROG_OF_WAR:
            observation = agent.AddFrogToObs(env, observation)  
        rewards = 0
        for s in range(30):
            action = agent.get_best_action(observation)
            new_observation, reward, terminated, truncated, info = env.step(action)
            new_observation = list(env.decode(new_observation))
            rewards += reward
            # env.render()
            observation = new_observation

            if terminated or truncated:
                break

    env.close()
    return

def show_data_from_file():
    """
    draw heatmap from file z_data
    """
    x_data = np.arange(1200,18000,600)
    y_data = np.arange(0.05, 1, 0.05)
    y_data = np.round(y_data, 2)
    z_data = np.loadtxt("z_data.txt", delimiter=",")
    # add30 and apply ReLU to each element
    z_data = ReLU(z_data+30)

    plt.imshow(z_data, cmap=plt.cm.hot, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(x_data)), x_data)
    plt.yticks(np.arange(len(y_data)), y_data)
    plt.xlabel("train times")
    plt.ylabel("learning rate")
    plt.title("train times - learning rate - average score")
    plt.show()
    return

def markov_agent_test():
    AGENT_TYPE = "markov_search"
    test_times = 1000
    FROG = True
    WEATHER = True
    train_times = 0

    scores = main.single_test(AGENT_TYPE,None,True,train_times,test_times,
                                  withWeather=WEATHER, FROG_OF_WAR=FROG)
    print("average score: ", np.mean(scores))
    print("stderror: ", np.std(scores) / np.sqrt(len(scores)))

    
    return

if True:
    show_performance()
    single_visualize()
    comp_different_agents()
    show_data_from_file()
    markov_agent_test()
    # single_rein_tests()