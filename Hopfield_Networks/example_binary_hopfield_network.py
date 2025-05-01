import numpy as np
import matplotlib.pyplot as plt


class HopfieldNetwork:
    def __init__(self):
        self.weights = None

    def train(self, patterns):
        num_neurons = patterns.shape[1]
        self.weights = np.zeros((num_neurons, num_neurons))

        for p in patterns:
            p = p.reshape(-1, 1)
            self.weights += np.dot(p, p.T)

        # Remove self-connections
        np.fill_diagonal(self.weights, 0)
        self.weights /= patterns.shape[0]

    def recall(self, pattern, steps=5):
        pattern = pattern.copy()
        for _ in range(steps):
            for i in range(len(pattern)):
                raw_input = np.dot(self.weights[i], pattern)
                pattern[i] = 1 if raw_input >= 0 else -1
        return pattern


# Create binary patterns to store (e.g., 3x3 grids)
def create_patterns():
    A = np.array([1, -1, 1,
                  -1, 1, -1,
                  1, -1, 1])

    B = np.array([-1, 1, -1,
                  1, -1, 1,
                  -1, 1, -1])

    return np.array([A, B])


def add_noise(pattern, noise_level=0.3):
    noisy = pattern.copy()
    flip = np.random.rand(*pattern.shape) < noise_level
    noisy[flip] *= -1
    return noisy


# Train and test the Hopfield Network
patterns = create_patterns()
network = HopfieldNetwork()
network.train(patterns)

test_pattern = add_noise(patterns[0], noise_level=0.3)

recalled = network.recall(test_pattern)


# Show original, noisy, and recalled patterns
def show_pattern(pattern, title):
    plt.imshow(pattern.reshape(3, 3), cmap='gray')
    plt.title(title)
    plt.axis('off')


plt.figure(figsize=(10, 3))
plt.subplot(1, 3, 1)
show_pattern(patterns[0], 'Original Pattern A')

plt.subplot(1, 3, 2)
show_pattern(test_pattern, 'Noisy Input')

plt.subplot(1, 3, 3)
show_pattern(recalled, 'Recalled Pattern')

plt.tight_layout()
plt.show()
