import numpy as np

class Agent(object):
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        return 
    
    def explore(self, observation):
        #virtual method
        pass

    def get_best_action(self, observation):
        #virtual method
        pass


class RandomAgent(Agent):
    def __init__(self, observation_space, action_space):
        super(RandomAgent, self).__init__(observation_space, action_space)
        return
    
    def explore(self, observation):
        raise Exception("random agent does not need training")
    
    def get_best_action(self, observation):
        return self.action_space.sample()


class ReinforcementAgent(Agent):
    def __init__(self, observation_space, action_space,
                 learning_rate=0.1, discount_factor=0.9, explore = 0.1):
        super(ReinforcementAgent, self).__init__(observation_space, action_space)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.explore_constant = explore
        # q_table: (observation, action, q_value/visit_times)
        self.q_table = np.zeros((self.observation_space.n, self.action_space.n, 2))
        #set visit time to 1
        self.q_table[:,:,1] = 1
        return 
    
    def explore(self, observation)->int:
        action = self.get_best_action(observation)
        self.q_table[observation, action, 1] += 1
        return action
    
    def update(self, old_observation, action, observation, reward):
        # Exploration Function
        # give the less visited action an additional weight
        next_actions = self.q_table[observation]
        next_actions = next_actions[:,0]+self.explore_constant/next_actions[:,1]
        temp = reward+self.discount_factor*np.max(next_actions)
        self.q_table[old_observation, action, 0] = \
            (1-self.learning_rate)*self.q_table[old_observation, action, 0] \
            + self.learning_rate*temp
        return

    def get_best_action(self, observation):
        # find the action with max q value in present state
        return np.argmax(self.q_table[observation, :, 0])