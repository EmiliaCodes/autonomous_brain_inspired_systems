import numpy as np
import random

# Grid parameters
grid_size = 11
num_actions = 4  # up, down, left, right

# Actions
actions = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1)    # right
}

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1
episodes = 1000

# Environment setup
goal_state =  (4, 7)

walls = set()
rows = grid_size
cols = grid_size

for i in range(rows):
    for j in range(cols):
        if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
            if (i, j) != goal_state:
                walls.add((i, j))

inside_walls = [(1, 2), (1, 4), (1, 6),
    (2, 2), (2, 4), (2, 6),
    (3, 2), (3, 4), (3, 6),
    (4, 2), (4, 4), (4, 6),
    (5, 2), (5, 4), (5, 6), (5, 7), (5, 8),
    (6, 2), (6, 4),
    (7, 2), (7, 4),
    (8, 2), (8, 4)]

walls.update(inside_walls)

# Rewards: -1 for every step, +10 for reaching goal
reward_matrix = np.full((grid_size, grid_size), -1)
reward_matrix[goal_state] = 10

# Initialize Q-table
Q = np.zeros((grid_size, grid_size, num_actions))

def is_valid_position(pos):
    return (0 <= pos[0] < grid_size and
            0 <= pos[1] < grid_size and
            pos not in walls)

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.choice(list(actions.keys()))  # explore
    else:
        return np.argmax(Q[state[0], state[1]])      # exploit



# Generate all possible grid elements
all_grid_elements = {(r, c) for r in range(grid_size) for c in range(grid_size)}

# Find the elements not in the set
elements_not_in_set = list(all_grid_elements - walls)

# Training
for episode in range(episodes):
    state = random.choice(elements_not_in_set)

    while state != goal_state:
        action = choose_action(state)
        move = actions[action]
        next_state = (state[0] + move[0], state[1] + move[1])

        if not is_valid_position(next_state):
            next_state = state  # can't move into walls or outside
            reward = -5         # optional: penalty for hitting wall
        else:
            reward = reward_matrix[next_state]

        old_value = Q[state[0], state[1], action]
        next_max = np.max(Q[next_state[0], next_state[1]])

        # Q-learning update
        Q[state[0], state[1], action] = old_value + alpha * (reward + gamma * next_max - old_value)

        state = next_state

# Display policy
policy = np.full((grid_size, grid_size), ' ')
arrow_map = ['↑', '↓', '←', '→']
for i in range(grid_size):
    for j in range(grid_size):
        if (i, j) == goal_state:
            policy[i, j] = 'G'
        elif (i, j) in walls:
            policy[i, j] = '█'
        else:
            best_action = np.argmax(Q[i, j])
            policy[i, j] = arrow_map[best_action]

print("Policy Grid (G = Goal, █ = Wall):")
for row in policy:
    print(' '.join(row))
