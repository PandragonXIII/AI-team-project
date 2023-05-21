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
    train_time_range = [0,4500]
    l_rate = 0.1
    d_factor = 0.99
    expl = 1

    y_data = []
    for train_times in range(train_time_range[0], train_time_range[1], 500):
        print("train times: ", train_times)
        scores = main.single_test(AGENT_TYPE,test_times,FROG_OF_WAR,train_times,l_rate,d_factor,expl, mute=True)
        avg_score = np.mean(scores)
        y_data.append(avg_score)
    x_data = range(train_time_range[0], train_time_range[1], 500)
    plt.plot(x_data, y_data)
    plt.xlabel("train times")
    plt.ylabel("average score")
    plt.title("train times - average score")
    plt.show()
    return

show_performance()