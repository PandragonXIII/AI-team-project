## 运行方法

### 0. 环境调整
安装gymnasium、numpy, 并用`modified_taxi.py`的内容覆盖/gymnasium/envs/toy_text/taxi.py文件。

### 1. 运行
运行`python3 main`

**Random Agent**: 将`main.py`第53行改为
```
scores = single_test("random")
```
可以用*test_times*参数改变测试样例数量

**Search Agent**: 将`main.py`第53行改为
```
scores = single_test("search")
```
可以用*test_times*参数改变测试样例数量

**Reinforcement Agent**: 将`main.py`第53行改为
```
scores = single_test("reinforcement")
```
将以默认参数运行
可选的参数：
*mute* 打印进度
*train_times* 训练次数
*test_times* 参数改变测试样例数量
*l_rate* 学习率
*d_factor* 衰减率$\gamma$ 
*expl* 探索率 
*visual* 展示测试过程

**Markov Agent**：将`main.py`第53行改为
```
scores = single_test("markov_search",)
```
//TODO


### 评估程序
运行`python3 statics_visual.py` 
在该文件中第205行开始，分别调用了六个函数：
- `show_performance()`：展示1200-15000学习次数、0.05-1学习率下强化学习Agent的得分（小于零分的记零分）。
- `single_visualize()`：展示(5)局强化学习后的游戏
- `comp_different_agents()`：以带误差的棒状图展示强化学习1650轮、2000轮以及搜索算法得分情况。
- `show_data_from_file()`：从`z_data.txt`中读取储存的数据并展示热力图
- `markov_agent_test()`：展示存在天气以及迷雾的情况下Markov Agent1000局游戏的average score & stderror。

以上函数的参数均可在函数内部（开头）修改
