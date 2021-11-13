
# imports
import numpy as np
import sys
import os

# config
SECTION = 3
MAX_CHOCOLATE = 20
MIN_CHOCOLATE = 3

VERBOSE=True

class GreedyChocolateGame:
    '''
    Greedy chocolate game class
    '''

    def __init__(self, SECTION, MIN_CHOCOLATE, MAX_CHOCOLATE, VERBOSE):

        # settings
        self.section = SECTION
        self.min_chocolate = MIN_CHOCOLATE
        self.max_chocolate = MAX_CHOCOLATE
        self.verbose = VERBOSE
        if VERBOSE:
            self.print_init() 
        self.reset()

    ################
    ## Game logic ##
    ################
    def reset(self):
        '''
        Reset function to restart the game, reinitialize the state
        '''

        # if self.verbose:
            # print("\nNew Game")

        # initialize chocolate boxes
        self.boxes = np.random.randint(
            low=self.min_chocolate, high=self.max_chocolate+1, size=self.section
        )

        self.player = "Player 1"

    def play(self, action=None):
        '''
        Play function
        '''

        if self.verbose:
            # Print current chocolates
            self.print_state()

            # Check if play versus human, action will be selected by human
            if action == None:

                # Ask for user's input
                box_choice = self.ask_box()
                chocolate_num = self.ask_chocolate(box_choice)

                # Get action list
                action = [box_choice, chocolate_num]

            # Notify turn
            self.print_take_chocolate(action[0], action[1])

        # Take the chocolate
        self.take_chocolate(action)

        # Check the winner
        done = self.check_greedy()

        # Notify winner
        if self.verbose and done:
            self.notify_winner()

        self.change_player()

        return done

    def take_chocolate(self, action):
        '''
        Function to take chocolate in a box
        '''
        self.boxes[action[0]] -= action[1]
    
    def check_greedy(self):
        '''
        Function to check if it's game over
        '''
        done = False
        if np.sum(self.boxes) == 0:
            done = True

        return done

    def change_player(self):
        '''
        Function to change player
        '''
        if self.player == "Player 1":
            self.player = "Player 2"
        else:
            self.player = "Player 1"

    ###################
    ## Print methods ##
    ###################
    def print_init(self):
        '''
        Print function for game instruction
        '''
        print("\n\n================================================================")
        print("\nWelcome to greedy test game!\n")
        print("\nPeople avoid taking the last chocolate because")
        print("it's a sign of greediness.\n\nYou can take any chocolates in a box")
        print("(as long as it's sufficient).\n")
        print("Just don't be the person to take the last chocolate!")
        print("\n================================================================")

    def print_state(self):
        '''
        Function to print the current chocolates in all boxes
        '''
        print("\n----------------------------------------------------------------")
        print("\n\nChocolate box:\n")
        for i in range(len(self.boxes)):
            print(f"Box {i+1}: {self.boxes[i]}")

    def print_take_chocolate(self, box_choice, num_chocolate):
        '''
        Function to print how many chocolate is taken
        '''
        print(f"\n{self.player}: take {num_chocolate} chocolates from box {box_choice+1}...")

    def notify_winner(self):
        '''
        Function to print the loser
        '''
        print(f"\n{self.player} take the last chocolate!")
        print(f"{self.player} is greedy :(")

    #################
    ## Interaction ##
    #################
    def ask_box(self):
        '''
        Function to ask which box to take
        '''

        while True:

            box_choice = input(f"\n{self.player} turn.\nPick a chocolate box: ")

            if box_choice == "q":
                sys.exit()

            try:
                box_choice = int(box_choice) - 1
            except:
                print(f"ERROR!! Please enter the box number from 1 to {self.section}\n")
                continue

            if box_choice+1 < 1 or box_choice+1 > self.section:
                print(f"ERROR!! Please enter the box number from 1 to {self.section}\n")
            elif self.boxes[box_choice] == 0:
                print(f"There's no more chocolate in box {box_choice+1}")
            else:
                break
        
        return box_choice

    def ask_chocolate(self, box_choice):
        '''
        Function to ask how many chocolate to take in a given box
        '''
        
        while True:
            # Ask how many chocolate to take
            num_chocolate = input(f"How many chocolates to take from box {box_choice+1}: ")
            if num_chocolate == "q":
                sys.exit()

            try:
                num_chocolate = int(num_chocolate)
            except:
                print("ERROR!! Please enter the an integer number!")
                continue

            if num_chocolate <= 0:
                print(f"Can't take {num_chocolate} chocolates")
            elif num_chocolate > self.boxes[box_choice]:
                print(f"Not enough chocolates, don't be greedy :(")
            else:
                break

        return num_chocolate

if __name__ == "__main__":

    # config
    SECTION = 3
    MAX_CHOCOLATE = 20
    MIN_CHOCOLATE = 3

    VERBOSE=True

    # vs agent
    if len(sys.argv) == 3:
        if sys.argv[1] == "vs_agent":
            if sys.argv[2] == "--random":
                # Create random agent
                from agent.base_agent import RandomAgent
                agent = RandomAgent()
            elif sys.argv[2] == "--best":
                # Create best agent
                from agent.td_agent import QLearningAgent
                agent = QLearningAgent(0, 0, 1)
                # Load model of the current best model
                filepath = os.path.join(
                    "model",
                    f"qagent_self_play_{SECTION}_{MIN_CHOCOLATE}_{MAX_CHOCOLATE}.npy"
                )
                agent.load_model(filepath)
            else:
                print("ERROR! vs_agent argument only have --random and --best parameters")
                exit(1)

            # Create new game
            game = GreedyChocolateGame(SECTION, MIN_CHOCOLATE, MAX_CHOCOLATE, VERBOSE)
            game.reset()

            # Set learning mode off
            agent.learning_mode_off()

            # Play the game
            while True:
                ################
                ## Human turn ##
                ################
                done = game.play()
                if done:
                    break
                ################
                ## Agent turn ##
                ################
                agent_action = agent.get_action(list(game.boxes))
                agent_action[0] -= 1
                done = game.play(agent_action)
                if done:
                    break

    else:
        print("\nPlaying 2 player games\n")
        game = GreedyChocolateGame(SECTION, MIN_CHOCOLATE, MAX_CHOCOLATE, VERBOSE)
        game.reset()

        while True:
            done = game.play()

            if done == True:
                break
