
# imports
import numpy as np
import sys
from game import GreedyChocolateGame

# config
SECTION = 3
MAX_CHOCOLATE = 20
MIN_CHOCOLATE = 3

class Environment:

    def __init__(self, SECTION, MIN_CHOCOLATE, MAX_CHOCOLATE):
        self.game = GreedyChocolateGame(SECTION, MIN_CHOCOLATE, MAX_CHOCOLATE, False)
        
    @property
    def state(self):
        return list(self.game.boxes)

    def get_possible_action(self):
        '''
        Function to returns all possible actions in a current state
        
        return possible_actions
        '''

        actions = []
        
        for box_num in range(len(self.state)):
            
            for choc_num in range(1, self.state[box_num] + 1):
                actions.append([box_num+1, choc_num])
        
        return actions

    def reset(self):
        '''
        Function to restart the state of game
        '''
        self.game.reset()
        return self.state

    def step(self, action):
        '''
        Function that get the result of action from the environment
        
        Input: action -> list of moves
        Output: next_state, reward, done
        '''
        action = action.copy()
        action[0] -= 1
        done = self.game.play(action)

        # winning, since the next player takes the last chocolate
        if np.sum(self.state) == 1:
            reward = 1
        # losing, taking the last chocolate
        elif np.sum(self.state) == 0:
            reward = -1
        # not winning nor losing
        else:
            reward = 0

        return self.state, reward, done


if __name__ == "__main__":

    env = Environment(SECTION, MAX_CHOCOLATE, MIN_CHOCOLATE)
    print(f"\nSample state (env.state): {env.state}")
    print("\nSample possible action (env.get_possible_action())" + 
        f": {env.get_possible_action()}")
    print(f"\nSample action taken (env.get_possible_action[0])" +
        f": {env.get_possible_action()[0]}")
    print(f"\nSample step using previous action (env.step(action))" +
        f"-> (next state, reward, done): {env.step(env.get_possible_action()[0])}")


