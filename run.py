
from maze_env import Maze
from DQN import DeepQNetwork
import copy
import numpy as np

def run_maze(agent):
    for episode in range(20000):
        print("Episode=" + str(episode))
        step = 0
        agent.restart()

        while True:
            
            location = np.copy(agent.location)

            # RL choose action based on observation
            action = RL.choose_action(location)
           
            # RL take action and get next observation and reward
            next_location, reward, terminal = agent.move(action)
            

            RL.store_transition(location, action, reward, next_location , terminal)
            
            if terminal or step > 4000:
                #print("step: " + str(step) + " reward: " + str(reward) )
                RL.learn()
                break;
           
            step += 1


if __name__ == "__main__":
    agent = Maze()
    RL = DeepQNetwork(agent.n_actions, agent.n_features)
    run_maze(agent)
    RL.test()
    
   
