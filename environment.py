import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools

# plt.rcParams['figure.figsize'] = [7,7]

'''
Environment class consists of the definition of the grid world.
'''

class Environment():
    def __init__(self, N = 10, dynamic_start = False, dynamic_goal = False, goal_loc = (10,10)):
        self.N = N
        self.dynamic_goal = dynamic_goal
        self.dynamic_start = dynamic_start
        self.reward = np.zeros((self.N+1, self.N+1))
        
        # Transition Probability
        self.trans_prob = {
                                'left': ['left', 'up', 'down'], 
                                'right': ['right', 'up', 'down'], 
                                'up': ['up','left', 'right'], 
                                'down': ['down', 'left', 'right']
                            }
        # Converting actions to numerical increments in the x-y co-ordinate system
        self.action_effect = {
                                'left': (-1, 0), 
                                'right': (1, 0), 
                                'up': (0, 1), 
                                'down': (0, -1)
                            }
        
        # Based on real co-ordinate system
        # Ablsolute x, y values
        self.wall_list = [
            (5,0), (5,2), (5,3),
            (5,4), (5,5), (5,6), 
            (5,7), (5,9), (5,10),
            (0,5), (2,5), (3,5), (4,5),
            (6,4), (7,4), (9,4), (10,4)
                    ]
        
        # layout_dict consists of absolute x, y as keys
        # [(inverted x, y for display) , value of the cell {np.nan if it is a valid cell, 1 if the cell is a wall}]
        self.layout_dict = dict()
        for x, y in itertools.product(range(self.N+1),range(self.N+1)):
            if (x,y) in self.wall_list:
                self.layout_dict[(x,y)] = [(y,x), 1]
            else:
                self.layout_dict[(x,y)] = [(y,x), np.nan]
        
        # generating data for grid world layout
        self.data = np.ones((self.N+1, self.N+1)) * np.nan
        for co_ord, val in self.layout_dict.values():
            self.data[co_ord] = val
        
        # To spawn the agent in any random legal grid. By default it spawns at (0,0)
        if self.dynamic_start:
            self.start_x = np.random.randint(1,self.N+1) - 0.5
            self.start_y = np.random.randint(1,self.N+1) - 0.5
            while (self.start_x,self.start_y) in self.wall_list:
                self.start_x = np.random.randint(1,self.N+1) - 0.5
                self.start_y = np.random.randint(1,self.N+1) - 0.5
        else:
            self.start_x = 0.5
            self.start_y = 0.5
        
        # To spawn the goal in any random legal grid. By default it spawns at (10,10)
        # Can also be obtained from user as input but not directly visible to the Agents
        if self.dynamic_goal:
            self.goal_x = np.random.randint(1,self.N+1) - 0.5
            self.goal_y = np.random.randint(1,self.N+1) - 0.5
            while (int(self.goal_y),int(self.goal_x)) in self.wall_list:
                self.goal_x = np.random.randint(1,self.N+1) - 0.5
                self.goal_y = np.random.randint(1,self.N+1) - 0.5
        else:
            self.goal_x = goal_loc[0] + 0.5
            self.goal_y = goal_loc[1] + 0.5
        
        self.reward[int(self.goal_x),int(self.goal_y)] = 1
    
        self.cur_pos = (int(self.start_x),int(self.start_y))
    
    # gets the current location of the agent in the grid world
    def currentPos(self):
        return self.cur_pos
    
    # get location of goal
    # generally not available to the agent
    def getGoalLoc(self):
        return (int(self.goal_x), int(self.goal_y))
    
    # When the agent takes an action and the resultant action 
    # due to uncertainity in the environment
    def takeAction(self, action):
        print('Action Executed: ', action)
        imp_action = np.random.choice(self.trans_prob[action], p=[0.8, 0.1, 0.1])
        print('Action Resulted: ', imp_action)
        return imp_action
    
    # To check if the action is valid. i.e, whether the actions leads 
    # the agent to a valid grid that is not a wall and exists
    def validAction(self,x_new, y_new):
        if (x_new < 0 or x_new > 10):
            print("Invalid Action!")
            return False
        if (y_new < 0 or y_new > 10):
            print("Invalid Action!")
            return False
        if (x_new, y_new) in self.wall_list:
            print("Invalid Action! It's a wall!")
            return False
        
        return True
    
    # Once the action is taken update the state of the agent
    # to the new location
    def updateState(self, action):
        x_cor = self.cur_pos[0]
        y_cor = self.cur_pos[1]
        
        take_action = self.action_effect[self.takeAction(action)]
        
        x_new = x_cor + take_action[0]
        y_new = y_cor + take_action[1]
        
        if self.validAction(x_new, y_new):
            print('Valid Action, Executing it.')
            self.cur_pos = (x_new, y_new)
        else:
            print('Invalid Action, Not-Executing it.')
        
        return self.cur_pos
    
    # Layout of the grid world
    # Can be displayed if required by the user
    # Prints necessary information about the agent-environment interaction
    def layout(self, plot = True, agent_loc = ()):
        if not agent_loc:
            agent_loc = (self.start_x,self.start_y)
        else:
            agent_loc = (agent_loc[0] + 0.5,agent_loc[1] + 0.5)
        print("Agent's Current Location: X - ", int(agent_loc[0]), " Y - ", int(agent_loc[1]))
        if plot:
            fig, ax = plt.subplots(1, 1, tight_layout=True)
            my_cmap = matplotlib.colors.ListedColormap(['grey'])
            my_cmap.set_bad(color='w', alpha=0)

            for x in range(self.N + 2):
                ax.axhline(x, lw=2, color='k', zorder=5)
                ax.axvline(x, lw=2, color='k', zorder=5)

            ax.imshow(np.flip(self.data, axis=0), interpolation='none', cmap=my_cmap, extent=[0, self.N+1, 0, self.N+1], zorder=0)

            ax.plot(agent_loc[0],agent_loc[1], '^', markersize = 25)
            ax.plot(self.goal_x,self.goal_y, '*', markersize = 25)

            ax.axis('off')
            plt.show()
    
    # get the reward from the environment
    def getReward(self, inp_state):
        return self.reward[inp_state]
    
    # check if the goal state has been reached by the agent
    def isGoal(self):
        return self.cur_pos == self.getGoalLoc()