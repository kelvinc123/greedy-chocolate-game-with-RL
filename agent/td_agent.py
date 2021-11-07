
from agent.base_agent import Agent
import numpy as np


class QLearningAgent(Agent):
    '''
    1-step off-policy Q learning agent
    '''
    
    def __init__(self, alpha, epsilon, discount):
        '''
        Initialize Q learning agent
        Input:
            alpha: Float : learning rate
                value between 0 and 1
            epsilon: Float : probability of taking random action for exploration
                value between 0 and 1
            discount: Float : discount rate of future reward
                value between 0 and 1
        '''
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount
        
    def set_epsilon(self, eps):
        '''
        Method to set epsilon

        Input:
            eps: Float: new epsilon
        '''
        self.epsilon = eps
        
    def get_value(self, state):
        '''
        Function to get value function 
        V(s) = max(Q(s, a))

        Input:
            state: array : current state
        Output:
            value: Float : value of a state
        '''
        actions = self._possible_actions(state)
        q_val = [self.get_qvalue(state, action) for action in actions]
        if len(q_val) == 0:
            return 0
        return np.max(q_val)
        
    def get_best_action(self, state):
        '''
        Get the best action to take with a given state
        returns the action with maximum q values

        Input:
            state: array : current state
        Output:
            action: array : best action
        '''
        actions = self._possible_actions(state)
        
        if len(actions) == 0:
            return None
        
        q_val = [self.get_qvalue(state, action) for action in actions]
        return actions[np.argmax(q_val)]
        
    def get_action(self, state):
        
        '''
        Method to get action given a current state

        Input:
            state: array : current state
        Output:
            action: array : action's taken
        '''

        # Get all possible actions
        actions = self._possible_actions(state)
        

        # If there are no legal actions, return None
        if len(actions) == 0:
            return None
        
        if self._learn:
            u = np.random.uniform()
            if u < self.epsilon:
                # Pick a random action for exploration
                chosen_action = actions[np.random.choice(range(len(actions)))]
            else:
                # Otherwise, get the best action from a given state
                chosen_action = self.get_best_action(state)
        else:
            # If not learning, just get the best action
            chosen_action = self.get_best_action(state)
        
        return chosen_action
    
    def update(self, state, action, reward, next_state):
        
        '''
        Function to update q value using the following formula
        Q(s,a) = (1 - alpha) * Q(s,a) + alpha * (r + gamma * V(s'))

        Input:
            state: array : current state
            action: array : action taken
            reward: Float : reward of taking action given current state
            next_state: array : next state after taking an action
        '''
        # update active only if the agent is learning
        if self._learn:

            new_value = (1 - self.alpha) * self.get_qvalue(state, action)
            new_value += (self.alpha * (reward + (self.discount * self.get_value(next_state))))
            
            self.set_qvalue(state, action, new_value)


class ExpectedSarsaAgent(QLearningAgent):

    '''
    1-step Expected Sarsa Agent, inherit QLearningAgent
    since they have lots of similarities. 
    '''

    def __init__(self, alpha, epsilon, discount):
        '''
        Initialize Q learning agent
        Input:
            alpha: Float : learning rate
                value between 0 and 1
            epsilon: Float : probability of taking random action for exploration
                value between 0 and 1
            discount: Float : discount rate of future reward
                value between 0 and 1
        '''
        super().__init__(alpha, epsilon, discount)

    def get_value(self, state):
        '''
        Override get_value function from QLearningAgent
        V(s) = E[Q(s, a)] = sum (p(a|s) * Q(s, a))

        Input:
            state: array : current state
        Output:
            value: Float : value of a state
        '''
        actions = self._possible_actions(state)
        q_val = [self.get_qvalue(state, action) for action in actions]
        if len(q_val) == 0:
            return 0
        
        # with probability epsilon, uniformly selecting action will result in
        # the same probability of all actions
        value = np.sum(q_val) * (self.epsilon / len(q_val))

        # with probability 1 - epsilon, select the best state
        value += (1 - self.epsilon) * self.get_qvalue(state, self.get_best_action(state))

        return value