import numpy as np


def AddFrogToObs(env, observation:list, visible_dis : int =2)->list:
    """
    passenger is visible only if it is within visible_dis
    return (taxi_row, taxi_col, passenger_location(None), destination(None))
    """
    taxi_row, taxi_col, passenger_location, destination = observation
    if passenger_location == 4: # passenger in taxi
        return observation
    else:
        passenger_row, passenger_col = env.locs[passenger_location]
        if abs(taxi_row - passenger_row) <= visible_dis and abs(taxi_col - passenger_col) <= visible_dis: #square visual
            return observation
        else:
            return [taxi_row, taxi_col, None, None]
        

class Agent(object):
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        return 
    
    def explore(self, observation):
        #virtual method
        pass

    def get_best_action(self, observation):
        #virtual method
        pass


class RandomAgent(Agent):
    def __init__(self, env):
        super(RandomAgent, self).__init__(env=env)
        return
    
    def explore(self, observation):
        raise Exception("random agent does not need training")
    
    def get_best_action(self, observation):
        return self.action_space.sample()


class ReinforcementAgent(Agent):
    def __init__(self, env,
                 learning_rate=0.1, discount_factor=0.9, explore = 0.1):
        super(ReinforcementAgent, self).__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.explore_constant = explore
        # q_table: (observation, action, q_value/visit_times)
        self.q_table = np.zeros((self.observation_space.n, self.action_space.n, 2))
        #set visit time to 1
        self.q_table[:,:,1] = 1
        return 
    
    def explore(self, observation:int|list)->int:
        #if observation is a tuple
        if not isinstance(observation, int):
            # print(observation)
            if None in observation:
                raise Exception("invisible passenger")
            else: # all locs are valid
                taxi_row, taxi_col, passenger_location, destination = observation
                observation = self.env.encode(taxi_row, taxi_col, passenger_location, destination)
        # do as usual (int)
        action = self.get_best_action(observation)
        self.q_table[observation, action, 1] += 1
        return action
    
    def update(self, old_observation:int|list, action, observation:int|list, reward):
        # Exploration Function
        # give the less visited action an additional weight
        if not isinstance(observation, int): #if observation is a list
            if None in observation:
                raise Exception("invisible passenger")
            else:
                taxi_row, taxi_col, passenger_location, destination = observation
                observation = self.env.encode(taxi_row, taxi_col, passenger_location, destination)
        if not isinstance(old_observation, int): #if observation is a list
            if None in old_observation:
                raise Exception("invisible passenger")
            else:
                taxi_row, taxi_col, passenger_location, destination = old_observation
                old_observation = self.env.encode(taxi_row, taxi_col, passenger_location, destination)
        next_actions = self.q_table[observation]
        next_actions = next_actions[:,0]+self.explore_constant/next_actions[:,1]
        temp = reward+self.discount_factor*np.max(next_actions)
        self.q_table[old_observation, action, 0] = \
            (1-self.learning_rate)*self.q_table[old_observation, action, 0] \
            + self.learning_rate*temp
        return

    def get_best_action(self, observation:int|list):
        # find the action with max q value in present state
        #if observation is iterable 
        if not isinstance(observation, int):
            if None in observation:
                raise Exception("invisible passenger")
            else:
                taxi_row, taxi_col, passenger_location, destination = observation
                observation = self.env.encode(taxi_row, taxi_col, passenger_location, destination)
        return np.argmax(self.q_table[observation, :, 0])