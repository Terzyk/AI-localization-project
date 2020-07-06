# prob.py
# This is

import random
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)
from gridutil import *

best_turn = {('N', 'E'): 'turnright',
             ('N', 'S'): 'turnright',
             ('N', 'W'): 'turnleft',
             ('E', 'S'): 'turnright',
             ('E', 'W'): 'turnright',
             ('E', 'N'): 'turnleft',
             ('S', 'W'): 'turnright',
             ('S', 'N'): 'turnright',
             ('S', 'E'): 'turnleft',
             ('W', 'N'): 'turnright',
             ('W', 'E'): 'turnright',
             ('W', 'S'): 'turnleft'}


class LocAgent:

    def __init__(self, size, walls, eps_perc, eps_move):
        self.size = size
        self.walls = walls
        # list of valid locations
        self.locations = list({*locations(self.size)}.difference(self.walls))
        # dictionary from location to its index in the list
        self.loc_to_idx = {loc: idx for idx, loc in enumerate(self.locations)}

        self.possible_dirs="NESW"

        self.list_of_states=list((loc[0],loc[1],direction) for loc in self.locations for direction in self.possible_dirs)

        print(self.list_of_states)
        self.state_dict = {pos: idx for idx,pos in enumerate(self.list_of_states)}
        print(self.state_dict)

        self.eps_perc = eps_perc
        self.eps_move = eps_move
        # previous action
        self.prev_action = None

        #self.P = np.ones([len(self.locations),4], dtype=np.float)
        self.P = np.ones([len(self.locations)*4],dtype=np.float)
    def __call__(self, percept):
        # update posterior
        # TODO PUT YOUR CODE HERE
        # macierz T (168x168)
        T=np.ones([len(self.locations)*4,len(self.locations)*4],dtype=np.float)
        if self.prev_action =="forward":
            for idx, loc in enumerate(self.list_of_states):
                next_loc = nextLoc((loc[0],loc[1]),loc[2])
                #commnted prints -> for checking if there is good indexes
                #print("curr loc is: " + str(next_loc) + " " + str("and my dir is: " + str(loc[2])))
                if legalLoc(next_loc,self.size) and (next_loc not in self.walls):
                    next_idx = self.state_dict[(next_loc[0],next_loc[1],loc[2])]
                    #for pp,oo in self.state_dict.items():
                        #if oo==next_idx:
                            #print('im going to: '+str(pp))
                    T[idx,next_idx]=1.0-self.eps_move
                    T[idx,idx]=self.eps_move
                else:
                    T[idx,idx]=1.0

        if self.prev_action =="turnleft":
            for idx, loc in enumerate(self.list_of_states):
                next_loc = nextLoc((loc[0],loc[1]),loc[2])
                if legalLoc(next_loc,self.size) and (next_loc not in self.walls):
                    next_idx = self.state_dict[(next_loc[0],next_loc[1],loc[2])]
                    T[idx,next_idx]=1.0-self.eps_move
                    T[idx,idx]=self.eps_move
                else:
                    T[idx,idx]=1.0

        if self.prev_action =="turnright":
            for idx, loc in enumerate(self.list_of_states):
                next_loc = nextLoc((loc[0],loc[1]),loc[2])
                if legalLoc(next_loc,self.size) and (next_loc not in self.walls):
                    next_idx = self.state_dict[(next_loc[0],next_loc[1],loc[2])]
                    T[idx,next_idx]=1.0-self.eps_move
                    T[idx,idx]=self.eps_move
                else:
                    T[idx,idx]=1.0

        #print(T)

        O = np.zeros([len(self.locations) * 4], dtype=np.float)
        for idx, loc in enumerate(self.list_of_states):
            percept2 = []
            if loc[2] == "N":
                if "right" in percept:
                    percept2.append('E')
                if "left" in percept:
                    percept2.append('W')
                if "fwd" in percept:
                    percept2.append('N')
                if "bckwd" in percept:
                    percept2.append('S')
            if loc[2] == "S":
                if "right" in percept:
                    percept2.append('E')
                if "left" in percept:
                    percept2.append('W')
                if "fwd" in percept:
                    percept2.append('S')
                if "bckwd" in percept:
                    percept2.append('N')
            if loc[2] == "W":
                if "right" in percept:
                    percept2.append('N')
                if "left" in percept:
                    percept2.append('S')
                if "fwd" in percept:
                    percept2.append('W')
                if "bckwd" in percept:
                    percept2.append('E')
            if loc[2] == "E":
                if "right" in percept:
                    percept2.append('S')
                if "left" in percept:
                    percept2.append('N')
                if "fwd" in percept:
                    percept2.append('E')
                if "bckwd" in percept:
                    percept2.append('W')
            prob = 1.0
            #print(percept2)
            #print("My percept in robot coordinates is: " + str(percept))
            #print("My percept in world coordinates is: " + str(percept2))
            #print("Position: " + str(loc))
            #print("#####")
            for d in ['N','E','S','W']:
                next_loc=nextLoc((loc[0],loc[1]),d)
                obstacle = (not legalLoc(next_loc,self.size)) or (next_loc in self.walls)
                if obstacle == (d in percept2):
                    prob = prob * (1-self.eps_perc)
                else:
                    prob = prob * self.eps_perc
                #print(str(prob)+" and loc is: "+str(loc)+" and there was obstacle? "+str(obstacle)+" and was there d in percept2? "+str((d in percept2)))
            O[idx]=prob
        print(O)




        #print(self.P)
        # -----------------------
        action = 'forward'
        # TODO CHANGE THIS HEURISTICS TO SPEED UP CONVERGENCE
        # if there is a wall ahead then lets turn
        if 'fwd' in percept:
            # higher chance of turning left to avoid getting stuck in one location
            action = np.random.choice(['turnleft', 'turnright'], 1, p=[0.8, 0.2])
        else:
            # prefer moving forward to explore
            action = np.random.choice(['forward', 'turnleft', 'turnright'], 1, p=[0.8, 0.1, 0.1])

        self.prev_action = action

        return action

    def getPosterior(self):
        # directions in order 'N', 'E', 'S', 'W'

        # put probabilities in the array
        # TODO PUT YOUR CODE HERE
        P_arr = np.zeros([self.size, self.size, 4], dtype=np.float)


        # -----------------------
        #print(P_arr)
        return P_arr

    def forward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    def backward(self, cur_loc, cur_dir):
        if cur_dir == 'N':
            ret_loc = (cur_loc[0], cur_loc[1] - 1)
        elif cur_dir == 'E':
            ret_loc = (cur_loc[0] - 1, cur_loc[1])
        elif cur_dir == 'W':
            ret_loc = (cur_loc[0] + 1, cur_loc[1])
        elif cur_dir == 'S':
            ret_loc = (cur_loc[0], cur_loc[1] + 1)
        ret_loc = (min(max(ret_loc[0], 0), self.size - 1), min(max(ret_loc[1], 0), self.size - 1))
        return ret_loc, cur_dir

    @staticmethod
    def turnright(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 1) % 4
        return cur_loc, dirs[idx]

    @staticmethod
    def turnleft(cur_loc, cur_dir):
        dir_to_idx = {'N': 0, 'E': 1, 'S': 2, 'W': 3}
        dirs = ['N', 'E', 'S', 'W']
        idx = (dir_to_idx[cur_dir] + 4 - 1) % 4
        return cur_loc, dirs[idx]
