import numpy as np

class Agent(object):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        return 
    
    def train(self, observation, reward):
        #virtual method
        pass

    def get_best_action(self, observation):
        #virtual method
        pass


class RandomAgent(Agent):
    def __init__(self, observation_space, action_space):
        super(RandomAgent, self).__init__(observation_space, action_space)
        return
    
    def train(self, observation, reward):
        raise Exception("random agent does not need training")
    
    def get_best_action(self, observation):
        return self.action_space.sample()


class ReinforcementAgent(Agent):
    def __init__(self, observation_space, action_space,
                 learning_rate=0.1, discount_factor=0.9):
        super(ReinforcementAgent, self).__init__(observation_space, action_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((self.observation_space.n, self.action_space.n))
        return 
    
    def train(self, observation, reward):
        pass

    def get_best_action(self, observation):
        return np.argmax(self.q_table, axis = observation)