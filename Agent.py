import numpy as np
from queue import Queue

def AddFrogToObs(env, observation:list, visible_dis : int = 1)->list:
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


class SearchAgent(Agent):
    def __init__(self, env):
        super(SearchAgent, self).__init__(env)
        self.search_path = None
        self.search_path_step = -1
        return
    
    def explore(self, observation):
        raise Exception("search agent does not need training")
    
    def get_best_action(self, observation):
        #if observation is iterable 
        if not isinstance(observation, int): # observation is tuple
            if None in observation:
                raise Exception("invisible passenger")
            else:
                taxi_row, taxi_col, passenger_location, destination = observation
                observation = self.env.encode(taxi_row, taxi_col, passenger_location, destination)
        else:
            taxi_row, taxi_col, passenger_location, destination = self.env.decode(observation)
        # do as usual (int)
        if passenger_location==4 and (taxi_row, taxi_col) == self.env.locs[destination]: #taxi at destination
            self.search_path = None
            self.search_path_step = -1
            return 5 #drop off
        if passenger_location!=4 and (taxi_row, taxi_col) == self.env.locs[passenger_location]: #taxi at passenger
            self.search_path = None
            self.search_path_step = -1
            return 4 #pick up
        if self.search_path is None:
            self.search_path = self.search(observation)
        self.search_path_step += 1
        return self.search_path[self.search_path_step]
    
    def search(self, observation: int):
        '''Based on breadth-first search'''
        taxi_row, taxi_col, passenger_location, destination = self.env.decode(observation)
        if passenger_location==4: #passenger in taxi
            goal=self.env.locs[destination]
        else:
            goal=self.env.locs[passenger_location]
        q=Queue()
        visited=set()
        taxiloc=(taxi_row, taxi_col)
        #print("@ ",taxiloc)
        #print("- ",goal)
        q.put((taxiloc,[]))
        visited.add(taxiloc)
        while not q.empty():
            csnode=q.get()
            currentstate=csnode[0]
            if currentstate==goal:
                return csnode[1]
            for s in self._getSuccessors(currentstate):
                if s[0] not in visited:
                    visited.add(s[0])
                    path=csnode[1].copy()
                    path.append(s[1])
                    q.put((s[0],path))
        print('NO PATH FOUND')
        return []
    
    def _getSuccessors(self,currentstate):
        ret=[]
        taxi_row, taxi_col = currentstate
        if taxi_row < 4:
            ret.append(((taxi_row + 1, taxi_col), 0)) # south
        if taxi_row > 0:
            ret.append(((taxi_row - 1, taxi_col), 1)) # north
        if taxi_col < 4 and self.env.desc[taxi_row + 1, 2 * taxi_col + 2] == b":":
            ret.append(((taxi_row, taxi_col + 1), 2)) # east
        if taxi_col > 0 and self.env.desc[taxi_row + 1, 2 * taxi_col] == b":":
            ret.append(((taxi_row, taxi_col - 1), 3)) # west
        return ret
    

class MarkovAgent(Agent):
    """
    Markov Agent can deal with fog of war, in which passenger is invisible
    use passenger position as evidence variable
        taxi knows there is a hidded variable (weather)
    that influences the passenger position distribution
        it can infer the passenger position distribution, 
    based on its observation of past positions
    """
    def __init__(self, env, discount_factor=0.9, explore_constant=0.1):
        super(MarkovAgent, self).__init__(env)
        self.explore_constant = explore_constant
        self.discount_factor = discount_factor
        self.q_table = np.zeros((self.env.observation_space.n, self.env.action_space.n, 2))
        self.q_table[:,:,1] = 1
        return
    
    def explore(self, observation):
        raise Exception("markov agent does not need training")
    
    def get_best_action(self, observation):
        #if observation is iterable 
        if not isinstance(observation, int): # observation is tuple
            if None in observation:
                raise Exception("invisible passenger")
            else:
                taxi_row, taxi_col, passenger_location, destination = observation
                observation = self.env.encode(taxi_row, taxi_col, passenger_location, destination)
        else:
            taxi_row, taxi_col, passenger_location, destination = self.env.decode(observation)
        # do as usual (int)
        if passenger_location==4 and (taxi_row, taxi_col) == self.env.locs[destination]: #taxi at destination
            return 5 #drop off
        if passenger_location!=4 and (taxi_row, taxi_col) == self.env.locs[passenger_location]: #taxi at passenger
            return 4 #pick up
        return self.get_best_action_markov(observation)
    
    def get_best_action_markov(self, observation):
        #if observation is iterable 
        if not isinstance(observation, int): # observation is tuple
            if None in observation:
                raise Exception("invisible passenger")
            else:
                taxi_row, taxi_col, passenger_location, destination = observation
                observation = self.env.encode(taxi_row, taxi_col, passenger_location, destination)
        else:
            taxi_row, taxi_col, passenger_location, destination = self.env.decode(observation)
        # do as usual (int)
        next_actions = self.q_table[observation]
        next_actions = next_actions[:,0]+self.explore_constant/next_actions[:,1]
        return np.argmax(next_actions)
    
    def update(self, old_observation, action, observation, reward):
        raise Exception("markov agent does not need training")
    
class MarkovSearchAgent(SearchAgent):
    def __init__(self, env):
        super(MarkovSearchAgent, self).__init__(env)
        self.prevlocs=[]
    
    def explore(self, observation):
        raise Exception("Markov agent does not need training")
    
    def get_best_action(self, observation):
        if not isinstance(observation, int): # observation is tuple
            taxi_row, taxi_col, passenger_location, destination = observation
            observation = self.env.encode(taxi_row, taxi_col, passenger_location, destination)
        else:
            taxi_row, taxi_col, passenger_location, destination = self.env.decode(observation)
        
        if 