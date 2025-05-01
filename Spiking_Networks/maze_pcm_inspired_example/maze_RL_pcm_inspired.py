"""
This is a simple maze-solving agent that uses a spiking neural network. Inspiration comes from the paper:

Bianchi, S., Muñoz-Martín, I., Hashemkhani, S., Pedretti, G. & Ielmini, D.
A bio-inspired recurrent neural network with self-adaptive neurons and PCM synapses for
solving reinforcement learning tasks. in vol. 00 1–5 (Institute of Electrical and Electronics Engineers Inc., 2020).

"""

import random
import numpy as np
import matplotlib.pyplot as plt


M = 10  # Maze size (10x10 for simplicity; set to 30 for full scale)

learning_rate = 0.2 # Learning rate for synaptic weight updates
directions = [(0,1), (1,0), (0,-1), (-1,0)]  # right, down, left, up


def in_bounds(x, y):
    """
    Checks if the provided coordinates (x, y) are within the bounds of the maze.

    :param x: The x-coordinate to check.
    :param y: The y-coordinate to check.
    :return: True if (x, y) is within bounds, False otherwise.
    """
    return 0 <= x < M and 0 <= y < M

class Neuron:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.threshold = 1.0 # Starting threshold
        self.potential = 0.0
        self.fired = False
        self.neighbors = []
        self.adaptation_rate = 0.3
        self.recovery_rate = 0.05
        self.synaptic_weights = {}
        self.is_wall = False

        self.penalty = 0.0
        self.reward = 0.0

    def connect_neighbors(self, grid):
        if self.is_wall:
            return
        for dx, dy in directions:
            nx, ny = self.x + dx, self.y + dy
            if in_bounds(nx, ny):
                neighbor = grid[nx][ny]
                self.neighbors.append(neighbor)
                self.synaptic_weights[neighbor] = random.uniform(0.05, 0.5)

    def receive_spike(self, weight):
        self.potential += weight

    def step(self):
        if self.potential >= self.threshold:
            self.fired = True
            self.potential = 0.0 # Reset the potential 
            self.threshold += self.adaptation_rate
            return True
        else:
            self.threshold = max(1.0, self.threshold - self.recovery_rate)
            self.fired = False
            return False

    def reset_synaptic_weights(self):
        for neighbor in self.synaptic_weights:
            self.synaptic_weights[neighbor] = random.uniform(0.05, 0.5)

    def strengthen_connection(self, neighbor):
        """
        Neurons that fire together, wire together.
        """
        if neighbor in self.synaptic_weights:
            self.synaptic_weights[neighbor] += learning_rate

class MazeNetwork:
    def __init__(self, size):
        self.size = size+1
        self.grid = [[Neuron(x, y) for y in range(self.size)] for x in range(self.size)]
        self.agent_start = (1, 1)
        self.agent_pos = self.agent_start
        self.goal_pos = (4, 7)
        self.walls = set()
        self.trial_path = []
        self.trial_path_map = np.zeros((self.size, self.size))
        self.spike_count = 0
        self.set_manual_walls()
        for row in self.grid:
            for neuron in row:
                neuron.connect_neighbors(self.grid)

    def set_manual_walls(self):
        rows = len(self.grid)
        cols = len(self.grid[0])

        for i in range(rows):
            for j in range(cols):
                if i == 0 or i == rows - 1 or j == 0 or j == cols - 1:
                    if (i, j) != self.agent_pos and (i, j) != self.goal_pos:
                        self.grid[i][j].is_wall = True
                        self.walls.add((i, j))

        wall_positions = [
            (1, 2), (1, 4), (1, 6),
            (2, 2), (2, 4), (2, 6),
            (3, 2), (3, 4), (3, 6),
            (4, 2), (4, 4), (4, 6),
            (5, 2), (5, 4), (5, 6), (5, 7), (5, 8),
            (6, 2), (6, 4),
            (7, 2), (7, 4),
            (8, 2), (8, 4),
        ]
        for x, y in wall_positions:
            if (x, y) != self.agent_pos and (x, y) != self.goal_pos:
                self.grid[x][y].is_wall = True
                self.walls.add((x, y))

    def reset_firing_and_potential(self):
        for row in self.grid:
            for neuron in row:
                neuron.fired = False
                neuron.potential = 0.0

    def get_random_start(self):
        candidates = [
            (x, y) for x in range(self.size) for y in range(self.size)
            if not self.grid[x][y].is_wall and (x, y) != self.goal_pos
        ]
        return random.choice(candidates)

    def reset_trial(self):
        self.agent_start = self.get_random_start()
        self.trial_path_map = np.zeros((self.size, self.size))
        self.trial_path = []
        self.spike_count = 0
        for row in self.grid:
            for neuron in row:
                if not neuron.is_wall and (neuron.x, neuron.y) != self.goal_pos:
                    neuron.reset_synaptic_weights()

    def apply_penalty(self, pos, is_wall=False):
        x, y = pos
        neuron = self.grid[x][y]
        penalty = 20.0 if is_wall else 2.0
        neuron.threshold += penalty
        neuron.penalty += penalty

    def apply_reward(self):
        for inx, pos in enumerate(self.trial_path[-20:]):
            x, y = pos
            neuron = self.grid[x][y]
            neuron.threshold = max(0.0, neuron.threshold - (1 * inx))
            neuron.reward += 0.1 * inx

    def step(self):
        x, y = self.agent_pos
        current_neuron = self.grid[x][y]

        for neighbor in current_neuron.neighbors:
            weight = current_neuron.synaptic_weights[neighbor]
            neighbor.receive_spike(weight)

        next_pos = None
        for neighbor in current_neuron.neighbors:
            if neighbor.step(): # If neighbor neuron fired
                if neighbor.is_wall:
                    self.apply_penalty((neighbor.x, neighbor.y), is_wall=True)
                    self.apply_penalty((current_neuron.x, current_neuron.y), is_wall=False)
                    break
                else: # neighbor.is not a wall
                    next_pos = (neighbor.x, neighbor.y)
                    self.trial_path_map[next_pos] += 1
                    self.trial_path.append((x, y))
                    fired_neighbor = neighbor
                    current_neuron.strengthen_connection(fired_neighbor)
                    self.reset_firing_and_potential()
                    break


        if next_pos:
            self.agent_pos = next_pos
            current_neuron.reset_synaptic_weights()

            if self.agent_pos == self.goal_pos:
                self.apply_reward()
                return True  # Goal reached

        return False  # Goal not reached

    def print_maze(self):
        for i in range(self.size):
            row = ""
            for j in range(self.size):
                if (i, j) == self.agent_pos:
                    row += "A "
                elif (i, j) == self.goal_pos:
                    row += "G "
                elif self.grid[i][j].is_wall:
                    row += "# "
                else:
                    row += ". "
            print(row)
        print("\n")

    def get_penalty_map(self):
        return np.array([[self.grid[i][j].penalty for j in range(self.size)] for i in range(self.size)])

    def get_threshold_map(self):
        return np.array([[self.grid[i][j].threshold for j in range(self.size)] for i in range(self.size)])

    def get_reward_map(self):
        return np.array([[self.grid[i][j].reward for j in range(self.size)] for i in range(self.size)])


# Example usage
maze_net = MazeNetwork(M)
num_trials = 250

for trial in range(num_trials):
    steps = 0
    while steps < 200:
        reached = maze_net.step()
        steps += 1
        if reached:
            print(f"Trial {trial+1}: Reached the goal in {steps} steps.")
            maze_net.reset_trial()
            break
    else:
        print(f"Trial {trial+1}: Did not reach the goal in 400 steps.")
        maze_net.reset_trial()

steps = 0
maze_net.reset_trial()
maze_net.agent_pos = (1, 1)
while steps < 250:
    reached = maze_net.step()
    # maze_net.print_maze()
    steps += 1
    if reached:
        print(f"Reached the goal in {steps} steps.")
        break
else:
    print(f"Did not reach the goal in 250 steps.")

print(maze_net.trial_path_map)
maze_net.print_maze()

# Visualize penalty and reward maps
penalty_map = maze_net.get_threshold_map()
reward_map = maze_net.get_reward_map()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Penalty Map")
plt.imshow(penalty_map, cmap='Reds', interpolation='nearest')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.title("Reward Map")
plt.imshow(reward_map, cmap='Greens', interpolation='nearest')
plt.colorbar()

plt.tight_layout()
plt.show()