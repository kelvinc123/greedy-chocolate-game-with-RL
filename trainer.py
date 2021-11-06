import numpy as np
from environment import MAX_CHOCOLATE, MIN_CHOCOLATE, Environment
from td_agent import QLearningAgent
from td_agent import ExpectedSarsaAgent
from base_agent import RandomAgent


class TwoPlayerGameTrainer:

    def play_and_train(self, env, agent1, agent2, n_games=1000, learn=True):
        '''
        Run a full game using two agents. Agent 1 can be set to learn, while agent 2
        can only play. This is necessary to make the environment fixed.
        
        Input:
            env: Environment object : environment of the game
            agent1: RL Agent : Instantiate of RL agent
            agent2: RL Agent : Instantiate of RL agent
            n_games: Integer : Number of games to play
            learn: Boolean : Indicates whether agent1 is learning
        Output:
            game_history: Array : Array with size n_games that is +1 if agent 1 wins, and -1 if
                                agent 1 loses
        '''
        # Define the common reward dictionary for two player game
        REWARD_DICT = {'WIN': 1, 'LOSE': -1, 'EXPLORE': 0}
        
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
            print(f"Running game {t+1}", end='\r')
            
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
                    # Update q table using winning reward and the next state of agent 2
                    agent1.update(s, a, REWARD_DICT['WIN'], next_s)
                    game_history[t] = 1
                    n_wins += 1
                    break
                else:
                    # Otherwise update q table using next state of agent 2
                    agent1.update(s, a, REWARD_DICT['EXPLORE'], next_s)
                    
                # next state
                s = next_s
                    
        print("\nDone!")
        return game_history, (n_wins/n_games)


if __name__ == "__main__":

    SECTION = 3
    MAX_CHOCOLATE = 20
    MIN_CHOCOLATE = 3

    env = Environment(SECTION, MAX_CHOCOLATE, MIN_CHOCOLATE)

    trainer = TwoPlayerGameTrainer()

    q_agent = QLearningAgent(alpha=0.1, epsilon=0.3, discount=1)
    es_agent = ExpectedSarsaAgent(alpha=0.1, epsilon=0.2, discount=1)
    random_agent = RandomAgent()

    game_history, win_rate = trainer.play_and_train(
        env, q_agent, random_agent, n_games=20000, learn=True)

    print(f"Win rate of Q agent vs Random in training mode: {win_rate}")

    print("\nPlay the game in battle mode...")
    game_history, win_rate = trainer.play_and_train(
        env, q_agent, random_agent, n_games=20000, learn=False)
    print(f"Win rate of Q agent vs Random in battle mode: {win_rate}")