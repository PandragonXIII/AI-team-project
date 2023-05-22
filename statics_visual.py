"""
visualize the profermance of the model under different parameters
"""
import matplotlib.pyplot as plt
import numpy as np
import main
import gymnasium as gym


def ReLU(x):
    return np.maximum(0, x)

def show_performance():
    AGENT_TYPE = "reinforcement"
    test_times = 50
    FROG_OF_WAR = False
    train_time_range = [4400,5000,400]
    l_rate = [0.1, 0.4]
    d_factor = 0.9
    expl = 1

    env = gym.make("Taxi-v3")
    

    x_data = np.arange(train_time_range[0], train_time_range[1], train_time_range[2])
    x_data = np.append([0], x_data)
    y_data = np.arange(l_rate[0], l_rate[1], 0.1)
    y_data = np.around(y_data, decimals=2)# 格式化y_data到两位小数
    z_data = []
    for l_rate in y_data:
        agent = main.Agent.ReinforcementAgent(env=env,
                    learning_rate=l_rate, discount_factor=d_factor, explore=expl)
        for idx in range(len(x_data)-1):
            train_times = x_data[idx+1]-x_data[idx]
            # train_times = x_data[idx+1]
            print("train times: ", x_data[idx+1])
            scores = main.single_test(AGENT_TYPE,agent,True,train_times,test_times)
            avg_score = ReLU(np.mean(scores))
            z_data.append(avg_score)
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
    agent = main.Agent.SearchAgent(env=env)
    train_times = 0
    scores = main.single_test('search',None,True,train_times,test_times,
                                False ,FROG_OF_WAR,l_rate,d_factor,expl)
    avg_score = ReLU(np.mean(scores))
    print("search agent score: ", avg_score)
    plt.show()

    return

# show_performance()

def single_rein_tests():
    AGENT_TYPE = "reinforcement"
    test_times = 50
    FROG_OF_WAR = False
    train_times = 4000
    l_rate = 0.02
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

single_rein_tests()

def comp_different_agents():
    test_times = 100
    FROG_OF_WAR = False
    train_times = 4000
    l_rate = 0.1
    d_factor = 0.99
    expl = 1

    x_data = ["random", "reinforcement", "search"]
    y_data = []
    for agent_type in x_data:
        scores = main.single_test(agent_type,None,True,train_times,test_times,
                                  False ,FROG_OF_WAR,l_rate,d_factor,expl)
        avg_score = ReLU(np.mean(scores))
        y_data.append(avg_score)
    print(y_data)
    # 绘制bar图
    plt.bar(x_data, y_data)
    plt.xlabel("agent type")
    plt.ylabel("exp2(average score)")
    plt.title("agent type - average score")
    plt.show()

    return

# comp_different_agents()