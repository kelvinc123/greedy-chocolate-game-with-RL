
from environment import MAX_CHOCOLATE, MIN_CHOCOLATE, Environment
from agent.td_agent import QLearningAgent
from agent.td_agent import ExpectedSarsaAgent
from agent.base_agent import RandomAgent

import matplotlib.pyplot as plt
import numpy as np
from IPython import display

class TwoPlayerGameTrainer:
    '''
    Class for two player games.

    Consists of:
        play -> play one game of agent vs agent
        play_and_train -> train agent by playing against other agent multiple times
        self_play -> train agent by playing against itself multiple times
    '''

    @staticmethod
    def play(env, agent1, agent2, reset=True, verbose=True):
        '''
        Agent 1 vs agent 2 play for observation
        Input:
            env: Environment object : environment of the game
            agent1: RL Agent : Instantiate of RL agent
            agent2: RL Agent : Instantiate of RL agent
            reset: Boolean : Indicates whether the environment is reset
            verbose: Boolean : Indicates whether the game info is printed
        Output:
            winner: Integer : 1 if agent 1 wins, otherwise -1
        '''

        agent1.learning_mode_off()
        agent2.learning_mode_off()

        # Print game output
        if verbose:
            env.game.verbose = True
        else:
            env.game.verbose = False

        # Reset environment
        if reset:
            s = env.reset()
        else:
            s = env.state

        done = False
        while True:

            ##################
            ## Agent 1 turn ##
            ##################
            a = agent1.get_action(s)  # Get action
            next_s, _, done = env.step(a)  # Step

            if done:
                return -1
            
            ##################
            ## Agent 2 turn ##
            ##################
            a = agent2.get_action(next_s)  # Get action
            next_s, _, done = env.step(a)  # Step

            if done:
                return 1

            s = next_s

    @staticmethod
    def play_and_train(env, agent1, agent2, n_games=10000, learn=True, verbose=True):
        '''
        Run a full game using two agents. Agent 1 can be set to learn, while agent 2
        can only play. This is necessary to make the environment fixed.

        Setting learn=True will update the q table and implement exploration with
        probability of epsilon
        
        Input:
            env: Environment object : environment of the game
            agent1: RL Agent : Instantiation of RL agent
            agent2: RL Agent : Instantiation of RL agent
            n_games: Integer : Number of games to play
            learn: Boolean : Indicates whether agent1 is learning
        Output:
            game_history: Array : Array with size n_games that is +1 if agent 1 wins,
                                and -1 if agent 1 loses
            win_rate: Float : Number of times agent 1 wins the game

        All agent must have the following method:
            agent.get_action(state) -> get action for a given state
            agent.update(state, action, reward, next_state) -> update qvalue for learning
        '''

        # Define the common reward dictionary for two player game
        REWARD_DICT = {'WIN': 1, 'LOSE': -1, 'EXPLORE': 0}
        env.game.verbose=False  # don't give any output
        
        game_history = np.zeros(n_games, dtype=np.int64)  # +1 for agent1, -1 for agent2
        n_wins = 0
        if learn:
            # Set learning mode on if learn = True
            agent1.learning_mode_on()
        else:
            # Otherwise set learning mode to off
            agent1.learning_mode_off()
            
        # Agent 2 will not learn
        agent2.learning_mode_off()

        # Loop until termination
        for t in range(n_games):
            if verbose:
                print(f"Running game: {t+1}", end='\r')
            
            # Get the initial state
            s = env.reset()
            done = False

            # Loop until game over
            while True:
                
                ##################
                ## Agent 1 turn ##
                ##################
                
                # Get action from agent 1
                a = agent1.get_action(s)
                
                # Make a step
                next_s, r, done = env.step(a)
                

                if done:
                    game_history[t] = -1
                    if learn:
                        # Update q table with losing reward
                        agent1.update(s, a, REWARD_DICT['LOSE'], next_s)
                    break
                
                ##################
                ## Agent 2 turn ##
                ##################
                
                # Get action from agent 2, using next_s from agent 1
                a2 = agent2.get_action(next_s)
                
                # Make a step
                next_s, r, done = env.step(a2)
                
                if done:
                    if learn:
                        # Update q table using winning reward and the next state of agent 2
                        agent1.update(s, a, REWARD_DICT['WIN'], next_s)
                    game_history[t] = 1
                    n_wins += 1
                    break
                else:
                    if learn:
                        # Otherwise update q table using next state of agent 2
                        agent1.update(s, a, REWARD_DICT['EXPLORE'], next_s)
                    
                # next state
                s = next_s
                    
        print("\nDone!")
        return game_history, (n_wins/n_games)

    @staticmethod
    def self_play(env, agent, n_games=10000, iteration=20, plot_output=True):
        '''
        Method to train agent by self play
        Input:
            env: Environment : Game environment
            agent: RL Agent : Instantiation of RL Agent
            n_games: Integer : Number of games each iteration
            iteration: Integer : Number of iteration
        Output:
            agent_new: RL Agent : New agent after self play training
        '''
    
        env.game.verbose = False  # Omit game output

        if plot_output:
            visualizer = Visualizer()
        
        import copy
        
        # Copy agent with its attributes
        agent_old = copy.deepcopy(agent)
        agent_old.learning_mode_off()
        
        agent_new = copy.deepcopy(agent)
        agent_new.learning_mode_on()
        
        win_rates = []

        for i in range(iteration):
            if not plot_output:
                print(f"\nIteration {i+1}")
            history, n_win = TwoPlayerGameTrainer.play_and_train(
                env=env, agent1=agent_new, agent2=agent_old, n_games = n_games, learn=True, verbose=False
            )

            # Record win rates
            win_rates.append(n_win)

            # Copy new agent attributes into to agent_old
            agent_old = copy.deepcopy(agent_new)

            # Notify winning rate
            if plot_output:
                visualizer.plot_win_rates(win_rates)
            else:
                print(f"Winning rate: {n_win}")

        return agent_new, win_rates


class Visualizer:
    
    @staticmethod
    def plot_win_rates(win_rates):
        _ = display.clear_output(wait=True)
        _ = display.display(plt.gcf())
        _ = plt.figure(figsize=(6, 4), dpi=80)
        _ = plt.clf()
        _ = plt.title('Self Play...')
        _ = plt.xlabel('Number of Games')
        _ = plt.ylabel('Score')
        _ = plt.plot(np.arange(1, len(win_rates)+1), win_rates)
        _ = plt.xlim(xmin=1, xmax=len(win_rates)+1)
        _ = plt.ylim(ymin=0)
        _ = plt.text(len(win_rates), win_rates[-1], str(win_rates[-1]))
        _ = plt.legend(["Winning rate"])
        _ = plt.grid()
        _ = plt.show()


if __name__ == "__main__":

    SECTION = 3
    MAX_CHOCOLATE = 20
    MIN_CHOCOLATE = 3

    env = Environment(SECTION, MAX_CHOCOLATE, MIN_CHOCOLATE)

    trainer = TwoPlayerGameTrainer()

    q_agent = QLearningAgent(alpha=0.1, epsilon=0.3, discount=1)
    esarsa_agent = ExpectedSarsaAgent(alpha=0.1, epsilon=0.2, discount=1)
    random_agent = RandomAgent()

    game_history, win_rate = trainer.play_and_train(
        env, q_agent, random_agent, n_games=20000, learn=True)

    print(f"Win rate of Q agent vs Random in training mode: {win_rate}")

    print("\nPlay the game in battle mode...")
    game_history, win_rate = trainer.play_and_train(
        env, q_agent, random_agent, n_games=20000, learn=False)
    print(f"Win rate of Q agent vs Random in battle mode: {win_rate}")