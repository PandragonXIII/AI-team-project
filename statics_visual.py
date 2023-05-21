"""
visualize the profermance of the model under different parameters
"""
import matplotlib.pyplot as plt
import numpy as np
import main

def show_performance():
    AGENT_TYPE = "reinforcement"
    test_times = 50
    FROG_OF_WAR = False
    train_time_range = [300,2300]
    l_rate = [0.05, 0.9]
    d_factor = 0.99
    expl = 1

    x_data = range(train_time_range[0], train_time_range[1], 300)
    y_data = np.arange(l_rate[0], l_rate[1], 0.05)
    y_data = np.around(y_data, decimals=2)# 格式化y_data到两位小数
    z_data = []
    for l_rate in y_data:
        for train_times in x_data:
            print("train times: ", train_times)
            scores = main.single_test(AGENT_TYPE,test_times,FROG_OF_WAR,train_times,l_rate,d_factor,expl, mute=True)
            avg_score = np.mean(scores)
            z_data.append(avg_score)
    # 绘制热力图
    z_data = np.array(z_data).reshape(len(y_data), len(x_data))
    plt.imshow(z_data, cmap=plt.cm.hot, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(len(x_data)), x_data)
    plt.yticks(np.arange(len(y_data)), y_data)
    plt.xlabel("train times")
    plt.ylabel("learning rate")
    plt.title("train times - learning rate - average score")
    plt.show()


    # plt.plot(x_data, z_data)
    # plt.xlabel("train times")
    # plt.ylabel("average score")
    # plt.title("train times - average score")
    # plt.show()
    return

show_performance()