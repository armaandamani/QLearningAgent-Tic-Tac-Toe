import numpy as np
import random
from tqdm import tqdm

# Create a new empty board
def newBoard():
    return np.zeros((3, 3), dtype=int)

# Check for a winner
def winner(board):
    for player in [1, -1]:
        # Check rows and columns
        for i in range(3):
            if np.all(board[i, :] == player) or np.all(board[:, i] == player):
                return player
        # Check diagonals
        if np.all(np.diag(board) == player) or np.all(np.diag(np.fliplr(board)) == player):
            return player
    # Check for a draw
    if not potentialActions(board):
        return 0  # Draw
    return None  # Game is ongoing

# Get available moves
def potentialActions(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

# Apply an action to the board
def takeAction(board, action, player):
    new_board = board.copy()
    new_board[action] = player
    return new_board

# Reward function
def reward(board, player):
    game_result = winner(board)
    if game_result == player:
        return 1  # Win
    elif game_result == -player:
        return -1  # Loss
    elif game_result == 0:
        return 0  # Draw
    return -0.1  # Small penalty for ongoing game

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def qValue(self, state, action):
        return self.q_table.get((state.tobytes(), action), 0.0)

    def actionChoice(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(potentialActions(state))
        q_values = {action: self.qValue(state, action) for action in potentialActions(state)}
        max_q = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == max_q]
        return random.choice(best_actions)

    def update_q_value(self, state, action, reward, next_state):
        max_q_next = max([self.qValue(next_state, a) for a in potentialActions(next_state)], default=0.0)
        current_q = self.qValue(state, action)
        new_q = current_q + self.alpha * (reward + self.gamma * max_q_next - current_q)
        self.q_table[(state.tobytes(), action)] = new_q

# Training function
def train(agent, episodes=50000):
    for _ in tqdm(range(episodes)):
        board = newBoard()
        player = 1
        state = board.copy()

        while True:
            action = agent.actionChoice(state)
            next_state = takeAction(state, action, player)
            reward_value = reward(next_state, player)
            agent.update_q_value(state, action, reward_value, next_state)

            if reward_value in [1, -1, 0]:  # Game over
                break

            state = next_state
            player *= -1  # Switch player

# Random opponent
def random_opponent(state):
    return random.choice(potentialActions(state))

# Testing function
def test(agent, games=10000):
    wins, draws, losses = 0, 0, 0

    for _ in tqdm(range(games)):
        board = newBoard()
        player = 1
        state = board.copy()

        while True:
            if player == 1:
                action = agent.actionChoice(state)
            else:
                action = random_opponent(state)

            state = takeAction(state, action, player)
            result = winner(state)

            if result == 1:
                wins += 1
                break
            elif result == -1:
                losses += 1
                break
            elif result == 0:
                draws += 1
                break

            player *= -1  # Switch player

    print(f"Wins: {wins}, Draws: {draws}, Losses: {losses}")

# Train and test
agent = QLearningAgent()
train(agent)
test(agent)
