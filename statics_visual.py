"""
visualize the profermance of the model under different parameters
"""
import matplotlib.pyplot as plt
import numpy as np
import main
import gymnasium as gym

def show_performance():
    AGENT_TYPE = "reinforcement"
    test_times = 50
    FROG_OF_WAR = False
    train_time_range = [800,4000,400]
    l_rate = [0.15, 0.71]
    d_factor = 0.99
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
            print("train times: ", train_times)
            scores = main.single_test(AGENT_TYPE,test_times,FROG_OF_WAR,train_times,
                                      l_rate,d_factor,expl, mute=True, agent=agent)
            avg_score = np.exp2(np.mean(scores))
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
    scores = main.single_test("search",test_times,FROG_OF_WAR,0,
                                l_rate,d_factor,expl, mute=True, agent=agent)
    avg_score = np.exp2(np.mean(scores))
    print("search agent score: ", avg_score)
    plt.show()

    return

show_performance()

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
        scores = main.single_test(agent_type,test_times,FROG_OF_WAR,train_times,l_rate,d_factor,expl, mute=True)
        avg_score = np.exp2(np.mean(scores))
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