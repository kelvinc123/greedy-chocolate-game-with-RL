from collections import defaultdict
import numpy as np

class Agent:
    '''
    Parent class for agents. All agents must inherit Agent class.
    '''
    
    def __init__(self):
        '''
        Initialize:
            qvalues : q values of all pair of states and actions
            learn : indicates if the agent is learning or not
        '''
        self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self._learn = True
        
    def learning_mode_on(self):
        '''
        Method to turn on learning mode
        '''
        self._learn = True
        
    def learning_mode_off(self):
        '''
        Method to turn off learning mode
        '''
        self._learn = False
        
    def get_qvalue(self, state, action):
        '''
        Method to get q value given state and action
        '''
        state=tuple(list(state))
        action=tuple(list(action))
        return self._qvalues[state][action]
    
    def set_qvalue(self, state, action, value):
        '''
        Function to set q value given state and action
        '''
        state=tuple(list(state))
        action=tuple(list(action))
        self._qvalues[state][action] = value
    
    def get_value(self, state):
        '''
        Function to get v value from a given state.
        Must be implemented on the child class
        '''
        raise NotImplementedError
        
    def update(self, state, action, reward, next_state):
        
        '''
        Method to learn.
        Must be implemented on the child class
        '''
        raise NotImplementedError
    
    def _possible_actions(self, state):
        '''
        Method to get possible actions in greedy chocolate game
        Input:
            state: array : current state
        Output:
            actions: array : All possible actions given current state
        '''
        actions = []
        
        for box_num in range(len(state)):
            
            for choc_num in range(1, state[box_num] + 1):
                actions.append([box_num+1, choc_num])
                
        return actions

    def save_model(self, filename):
        '''
        Method to save q value from a file
        '''
        pass

    def load_model(self, filename):
        '''
        Method to load q value from a file
        '''
        pass

class RandomAgent(Agent):
    
    '''
    Random agent that behaves randomly.
    '''
    
    def __init__(self):
        super().__init__()
        
    def get_value(self, state):
        '''
        Method to get value on a given state
        '''
        # Get all actions
        actions = self._possible_actions(state)

        # Get all q value of all state and actions
        q_val = [self.get_qvalue(state, action) for action in actions]

        if len(q_val) == 0:
            return 0

        # Expected q value of taking action randomly
        return np.sum(q_val) / len(q_val)
        
    def get_action(self, state):
        
        '''
        Method to get action given state
        '''
        
        # Get all possible actions
        actions = self._possible_actions(state)
        
        # Return none if no valid action
        if len(actions) == 0:
            return None
        
        # Pick an action randomly
        idx_choice = np.random.choice(range(len(actions)))
        
        return actions[idx_choice]
    
    def update(self, state, action, reward, next_state):
        pass