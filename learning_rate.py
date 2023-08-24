import matplotlib.pyplot as plt
import numpy as np


def decayed_learning_rate(step, initial_learning_rate, decay_rate, decay_steps):
    return initial_learning_rate * (decay_rate ** (step / decay_steps))


learning_rate = 0.1
decay_rate = 0.9
decay_steps = 100

steps = np.arange(1, decay_steps + 1, 1)
learning_rates = []

for step in steps:
    learning_rate = decayed_learning_rate(step, learning_rate, decay_rate, decay_steps)
    learning_rates.append(learning_rate)

print(learning_rate)

lr = 0.1

for step in steps:
    lr = lr * decay_rate ** (step / decay_steps)
    print(lr)

plt.plot(steps, learning_rates)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("Decayed Learning Rate")

plt.grid(True)
plt.show()
