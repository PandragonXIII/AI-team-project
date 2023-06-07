import numpy as np
from queue import Queue, PriorityQueue

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
        # if abs(taxi_row - passenger_row)+abs(taxi_col - passenger_col) <= visible_dis: #manhatton visual
            return observation
        else:
            return [taxi_row, taxi_col, None, None]
        

class Agent(object):
    def __init__(self, env):
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.env = env
        return 
    
    def setup(self): # one-time operations at start of each case
        #virtual method
        pass

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
    
    def setup(self):
        pass

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
    
    def setup(self):
        pass

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
    
    def setup(self):
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
        #print("path step:",self.search_path_step,"searched action:",self.search_path[self.search_path_step])
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
    
    
class MarkovSearchAgent(SearchAgent):
    """
    MarkovSearchAgent can deal with fog of war, in which passenger is invisible
    use passenger position as evidence variable
        taxi knows there is a hidded variable (weather)
    that influences the passenger position distribution
        it can infer the passenger position distribution, 
    based on its observation of past positions
    It goes straight towards the most likely passenger location, and if no passenger there, goes straight for the next, etc.
    """
    def __init__(self, env, WEATHER_TRANSITION, PASSENGER_LOC_PROB):
        super(MarkovSearchAgent, self).__init__(env)
        self.prevloc = None
        self.pathstochoose = PriorityQueue()
        self.blinded = True
        self.WEATHER_TRANSITION = WEATHER_TRANSITION
        self.PASSENGER_LOC_PROB = PASSENGER_LOC_PROB
        self.lenbetweenlocs=[[0,8,4,7],
                            [8,0,8,5],
                            [4,8,0,7],
                            [7,5,7,0]] #lengths of paths between stations
        self.pLastweather_prevlocs = np.array([1/3,1/3,1/3])
        self.pThisweather_locs = np.full((4,3),1/3)
    
    def setup(self):
        self.pathstochoose = PriorityQueue()
        self.blinded = True
        self.filtering()

    def explore(self, observation):
        raise Exception("markov_search agent does not need training")
    
    def get_best_action(self, observation):
        if not isinstance(observation, int): # observation is tuple
            taxi_row, taxi_col, passenger_location, destination = observation
        else:
            taxi_row, taxi_col, passenger_location, destination = self.env.decode(observation)
        #print(taxi_row, taxi_col)
        if passenger_location is None: # can't see passenger
            #print("can't see passenger")
            self.blinded = True
            if self.search_path is None:
                self.choosePath(taxi_row, taxi_col)
                self.search_path_step = -1
            else:
                if len(self.search_path) - self.search_path_step <= 2: # near the loc but no passenger there
                    #print("near loc but no passenger there")
                    self.choosePath(taxi_row, taxi_col)
                    self.search_path_step = -1
            self.search_path_step += 1
            #print(self.search_path[self.search_path_step])
            return self.search_path[self.search_path_step]
        else:
            #print("saw passenger at loc",passenger_location)
            if self.blinded == True:
                self.search_path = None
                self.search_path_step = -1
            self.blinded = False
            if passenger_location < 4:
                self.prevloc = passenger_location
            elif (taxi_row, taxi_col) == self.env.locs[destination]: # crystallize upon arrival
                self.pLastweather_prevlocs = self.pThisweather_locs[self.prevloc]
                print("Today's weather probabilities:",self.pLastweather_prevlocs)
            return super(MarkovSearchAgent, self).get_best_action(observation)

    def choosePath(self, taxi_row, taxi_col):
        e_arrival=[15.25, 14.75, 15.25, 15.25] # The expected rewards of sending a passenger to destination, starting from passenger's departure
        # e_arrival[starting_loc] = 20 - \sigma_{loc}{manhattan distances to each loc from starting_loc} / num_locs
        if not self.prevloc: #if no info available, choose the current shortest path
            problist = np.array([.25, .25, .25, .25])
        else: # choose path according to distance and probability
            problist = self.calculateProb()
        print("passenger probs:",problist)
        if self.pathstochoose.empty():
            for loc in range(4): # search a path for all 4 locs
                cpath=self.search(self.env.encode(taxi_row, taxi_col, loc, 0))
                self.pathstochoose.put([-self._calculateExp(len(cpath),loc,range(4),problist,e_arrival), cpath, loc]) #actually calculates expected reward of each loc
        else:
            leftlocs=[]
            while not self.pathstochoose.empty(): #empty and rebuild self.pathstochoose
                leftlocs.append(self.pathstochoose.get()[2])
            for loc in leftlocs: # search a path for all #4 locs
                cpath=self.search(self.env.encode(taxi_row, taxi_col, loc, 0))
                self.pathstochoose.put([-self._calculateExp(len(cpath),loc,leftlocs,problist,e_arrival), cpath, loc]) #actually calculates expected reward of each loc
        val, self.search_path, headfor = self.pathstochoose.get()
        if len(self.search_path)<=2: # near the loc but no passenger there
            val, self.search_path, headfor = self.pathstochoose.get() # get path to another loc
        #print("Path chosen:",self.search_path,", going for loc",headfor)
        print("going for loc",headfor)

    def _calculateExp(self,lenpath,loc,loclist,problist,e_arrival):
        loclist=list(loclist)
        if len(loclist)<=1: return e_arrival[loc] - lenpath #only possibility
        ret = problist[loc] * (e_arrival[loc] - lenpath) #considering case if pass at loc
        newproblist=problist.copy() #considering case if pass not at loc: renew problist
        newproblist[loc]=0
        newproblist /= np.sum(newproblist)
        newloclist=loclist.copy()#renew loclist
        newloclist.remove(loc)
        en=[0,0,0,0] #expected rewards of other locs given passenger not at this loc
        for l in newloclist:
            en[l] = self._calculateExp(self.lenbetweenlocs[loc][l], l, newloclist, newproblist, e_arrival) - self.lenbetweenlocs[loc][l]
        ret += (1-problist[loc]) * np.max(np.array(en))
        return ret

    def calculateProb(self):
        #return np.array([.25, .25, .25, .25])
        pThisweather_prevlocs = np.dot(self.pLastweather_prevlocs, self.WEATHER_TRANSITION)
        ret = np.dot(pThisweather_prevlocs, self.PASSENGER_LOC_PROB)
        #print(ret)
        return ret

    def filtering(self):
        '''calculate probabilities of weather by Filtering algorithm'''
        pThisweather_prevlocs = np.dot(self.pLastweather_prevlocs, self.WEATHER_TRANSITION)
        for i in range(4):
            self.pThisweather_locs[i] = self.PASSENGER_LOC_PROB[:,i] * pThisweather_prevlocs
            self.pThisweather_locs[i] /= np.sum(self.pThisweather_locs[i]) # normalize 
