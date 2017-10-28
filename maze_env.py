"""
Reinforcement learning to walk maze
Candy reward +1
Bomb  reward -1

The RL is in RL_brain.py.
"""
import numpy as np
import time
import sys
from math import *
import random
import time

MAZE_H = 4  # grid height
MAZE_W = 4  # grid width


class Maze():
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        
        # For a 2D map, first I take x and y coordinate as feature
        # to high bias i.e. (0,0) always output different result as (3,3)
        # 2D map use one-hot encoding as a feature 
        self.n_features = MAZE_H*MAZE_W
        self.candy_loc = 9
        self.bomb_loc = 6
        self.location = np.zeros([1, MAZE_H*MAZE_W])
        self.restart()

    # restart the starting point
    def restart(self):    
        self.location = np.zeros([1, MAZE_H*MAZE_W])
        self.location[0,0] = 1
        
        # Random initialize the starting point
        '''
        self.location[0,int(floor(np.random.uniform(0, MAZE_H*MAZE_H)))] = 1
        state = np.where(self.location == 1)
        state = int(state[1])
        if  state == 9:
            self.reset()
        '''

    # move to next location given the action
    def move(self, action):
        # find the current location
        state = np.where(self.location == 1)
        state = int(state[1])
        
        # determine moving to the boundary
        bumping_wall = False
        
        # move to next location given action
        if action == 0:   # up
            if state > MAZE_H:
                self.location[0,state]=0
                self.location[0,state-MAZE_W]=1
            else:
                bumping_wall = True
        elif action == 1:   # down
            if state < MAZE_W*(MAZE_H-1):
                self.location[0,state]=0
                self.location[0,state+MAZE_W]=1
            else:
                bumping_wall = True
        elif action == 2:   # left
            if state%4 != 0:
                self.location[0,state]=0
                self.location[0,state-1]=1
            else:
                bumping_wall = True
        elif action == 3:   # right
            if (state+1)%4 != 0:
                self.location[0,state]=0
                self.location[0,state+1]=1
            else:
                bumping_wall = True

        
        # reward function
        state = np.where(self.location == 1)
        state = int(state[1])
        
        # if get candy, reward >0
        # if get bomb or bump to wall, reward < 0
        
        if state == 9:
            reward = 1
            terminal = 1
        else:
            reward = 0
            '''
            if bumping_wall == True:
                reward = -0.5
            '''
            terminal = 0
        
        next_location = np.copy(self.location)
        return next_location, reward, terminal

