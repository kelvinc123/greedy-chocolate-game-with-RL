
# imports
import numpy as np
import sys

# config
SECTION = 3
MAX_CHOCOLATE = 20
MIN_CHOCOLATE = 3

VERBOSE=True

class GreedyChocolateGame:

    def __init__(self, SECTION, MAX_CHOCOLATE, MIN_CHOCOLATE, VERBOSE):

        # setting 
        self.section = SECTION
        self.max_chocolate = MAX_CHOCOLATE
        self.min_chocolate = MIN_CHOCOLATE
        self.verbose = VERBOSE
        self.reset()


    def reset(self):
        '''
        Reset function to restart the game
        '''

        # initialize chocolate boxes
        self.boxes = np.random.randint(
            low=self.min_chocolate, high=self.max_chocolate+1, size=self.section
        )

        if VERBOSE:
            self.print_init()

        self.turn = "Player 1"

    def play(self, action=None):
        '''
        Play function
        '''

        if VERBOSE and not action:

            # Print current chocolates
            self.print_state()

            # Ask for user's input
            box_choice = self.ask_box()

            chocolate_num = self.ask_chocolate(box_choice)

            self.print_take_chocolate(box_choice, chocolate_num)

            action = [box_choice, chocolate_num]

        # take the chocolate
        self.take_chocolate(action)

        # check the winner
        done = self.check_greedy()

        if VERBOSE and done:
            self.notify_winner()

        return done

    def print_init(self):
        '''
        Print function for game instruction
        '''
        print("\n\n================================================================")
        print("\nWelcome to greedy test game!\n");
        print("\nPeople avoid taking the last chocolate because");
        print("it's a sign of greediness.\n\nYou can take any chocolates in a box");
        print("(as long as it's sufficient).\n");
        print("Just don't be the person to take the last chocolate!");
        print("\n================================================================")

    def ask_box(self):
        '''
        Function to ask which box to take
        '''

        while True:

            box_choice = input(f"\n{self.turn} turn.\nPick a chocolate box: ")

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


    def print_state(self):
        '''
        Function to print the current chocolates in all boxes
        '''

        print("\n\n\nChocolate box:\n")
        for i in range(len(self.boxes)):
            print(f"Box {i+1}: {self.boxes[i]}")

    def print_take_chocolate(self, box_choice, num_chocolate):
        '''
        Function to print how many chocolate is taken
        '''
        print(f"\nTaking {num_chocolate} chocolates from box {box_choice+1}...")

    def notify_winner(self):
        '''
        Function to print the loser
        '''
        print("\nYou take the last chocolate!")
        print(f"{self.turn} is greedy :(")

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
        if self.turn == "Player 1":
            self.turn = "Player 2"
        else:
            self.turn = "Player 1"

if __name__ == "__main__":

    game = GreedyChocolateGame(SECTION, MAX_CHOCOLATE, MIN_CHOCOLATE, VERBOSE)
    game.reset()

    while True:
        done = game.play()

        if done == True:
            break

        game.change_player()
