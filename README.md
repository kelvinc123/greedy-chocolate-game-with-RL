# Reinforcement Learning Agent for Greedy Chocolate Game

**Game Instruction:**
Given some boxes of chocolates. Each turn a player must pick a box and take at least one chocolate in that boxes. Player
can't take chocolate more than the amount of chocolate in a chosen box. Whoever pick the last chocolate is greedy (losing the game).

### Play the game:
The game is currently set with 3 boxes of chocolates. Each box has a random number of chocolates between 3 and 20.
 * 2 players: python game.py
 * vs random agent: python vs_agent --random
 * vs best agent: python vs_agent --best

### Environment:
The environment class is available in environment.py. 

Methods:
```
state -> Represents current state which is the number of chocolates on each box
reset -> Reset the game and return current state
step -> Take action as input, return next_state, reward, and done
get_possible_action -> Method to get possible actions in a current state
```

### Agent:
 * Random Agent
 * Tabular QLearning Agent
 * Tabular Expected Sarsa Agent

To view the training process, see train.ipynb

**Future instruction coming soon**
