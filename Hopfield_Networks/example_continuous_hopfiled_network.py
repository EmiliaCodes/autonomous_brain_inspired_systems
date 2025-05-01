import numpy as np
import matplotlib.pyplot as plt


class ContinuousHopfieldNetwork:
    def __init__(self, num_neurons, tau=1.0, dt=0.1, gain=5.0):
        self.num_neurons = num_neurons
        self.tau = tau  # Time constant
        self.dt = dt  # Time step
        self.gain = gain  # Steepness of activation function
        self.weights = np.zeros((num_neurons, num_neurons))
        self.u = np.zeros(num_neurons)  # Membrane potentials

    def train(self, patterns):
        """Hebbian learning rule"""
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= len(patterns)

    def activation(self, u):
        """Continuous activation function (tanh-like sigmoid)"""
        return np.tanh(self.gain * u)

    def recall(self, input_pattern, steps=50):
        self.u = input_pattern.copy()
        for _ in range(steps):
            du = (-self.u + self.weights @ self.activation(self.u)) / self.tau
            self.u += self.dt * du
        return self.activation(self.u)


# Create continuous patterns
def create_patterns():
    A = np.array([1, -1, 1,
                  -1, 1, -1,
                  1, -1, 0])

    B = np.array([-1, 0.8, -1,
                  0.8, -1, 0.8,
                  -1, 0.8, -1])

    return np.array([A, B])


# Add noise
def add_noise(pattern, noise_level=0.3):
    noise = np.random.uniform(-1, 1, pattern.shape) * noise_level
    return pattern + noise


# Train and test the CHN
patterns = create_patterns()
network = ContinuousHopfieldNetwork(num_neurons=9)
network.train(patterns)

noisy_input = add_noise(patterns[0], noise_level=0.5)
recalled = network.recall(noisy_input)


# Visualize
def show_pattern(pattern, title):
    plt.imshow(pattern.reshape(3, 3), cmap='gray', vmin=-1, vmax=1)
    plt.title(title)
    plt.axis('off')


plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
show_pattern(patterns[0], 'Original Pattern A')

plt.subplot(1, 3, 2)
show_pattern(noisy_input, 'Noisy Input')

plt.subplot(1, 3, 3)
show_pattern(recalled, 'Recalled Pattern')

plt.tight_layout()
plt.show()
