import numpy as np
import random
from tqdm import tqdm
import sys

# Create the board
def newBoard():
    return np.zeros((3, 3), dtype=int)

def reward(board, player):
    "Returns 1 if the player wins, -1 if the player loses, 0 for a draw, and -0.1 for an ongoing game."
    game_result = winner(board)
    if game_result == player:
        return 1
    elif game_result == -player:
        return -1
    elif game_result == 0:
        return 0
    return -0.1  # negative reward for ongoing game


def potentialActions(board):
    "Return a list of all possible actions (empty positions) on the board."
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]


def takeAction(board, action, player):
    "Apply an action (move) to the board for a given player."
    board[action] = player
    return board


# Check for a winner
def winner(board):
    "Returns 1 if player 1 wins, -1 if player -1 wins, 0 if it's a draw, and None if the game is ongoing."
    
    for player in [1, -1]:
        # Check rows and columns for a win
        for i in range(3):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return player
        # Check diagonals for a win
        if board[0, 0] == board[1, 1] == board[2, 2] == player:
            return player
        if board[0, 2] == board[1, 1] == board[2, 0] == player:
            return player
    # Check for a draw
    if np.all(board != 0):
        return 0  # Draw
    return None  # No winner yet

class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        "Initialize the Q-learning agent with given alpha, gamma, and epsilon."
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def qValue(self, state, action):
        "Get the Q-value for a given state and action."
        return self.q_table.get((state.tobytes(), action), 0.0)

    def actionChoice(self, state):
        "Choose an action based on the epsilon-greedy policy."
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(potentialActions(state))
        else:
            q_values = {action: self.qValue(state, action) for action in potentialActions(state)}
            max_q = max(q_values.values())
            actions_with_max_q = [action for action, q_value in q_values.items() if q_value == max_q]
            return random.choice(actions_with_max_q)

    def update_q_value(self, state, action, reward, next_state):
        "Update the Q-value for a given state-action pair using the Q-learning update rule."
        max_q_next = max([self.qValue(next_state, a) for a in potentialActions(next_state)], default=0.0)
        current_q = self.qValue(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)
        self.q_table[(state.tobytes(), action)] = new_q

def train(agent, episodes = 1000):
    "Train the Q-learning agent over a specified number of episodes."
    for episode in tqdm(range(episodes)):
        board = newBoard()
        player = 1  # Start with player 1
        state = board.copy()
        
        for step in range(9):  # Max steps in a game is 9
            action = agent.actionChoice(state)
            nextState = takeAction(board.copy(), action, player)
            rewardValue = reward(nextState, player)
            agent.update_q_value(state, action, rewardValue, nextState)

            # Debugging: print current episode, player, and action
            print(f"Episode: {episode}, Step: {step}, Player: {player}, Action: {action}, Reward: {rewardValue}")
            print(f"Board state:\n{nextState}")

            if rewardValue in [1, -1, 0]:  # End of the game
                print(f"Game ended with reward: {rewardValue}")
                break
            
            state = nextState
            player *= -1  # Switch player

            # Safety check: ensure the loop does not go on forever
            if step == 8:
                print("Reached maximum steps in the game.")
                break

def random_opponent(state):
    "Choose a random action for the opponent."
    return random.choice(potentialActions(state))

def test(agent, games=10000):
    "Test the Q-learning agent over a specified number of games."
    
    wins = 0
    draws = 0
    losses = 0
    for game in tqdm(range(games)):
        board = newBoard()
        player = 1
        state = board.copy()
        
        for step in range(9):  # Max steps in a game is 9
            if player == 1:
                action = agent.actionChoice(state)
            else:
                action = random_opponent(state)
            
            nextState = takeAction(state.copy(), action, player)
            rewardValue = reward(nextState, player)
            
            if rewardValue == 1 and player == 1:
                wins += 1
                break
            elif rewardValue == 1 and player == -1:
                losses += 1
                break
            elif rewardValue == 0 and winner(nextState) == 0:
                draws += 1
                break
            
            state = nextState
            player *= -1  # Switch player
    
    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")


agent = QLearningAgent()
train(agent)
test(agent)
