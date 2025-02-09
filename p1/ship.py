import random
import numpy as np

# Set or change D
D = 5

# 0 = blocked cell, 1 = open cell
ship = np.zeros((5, 5))

# Random cell to open
ship[random.randint(0, D-1), random.randint(0, D-1)] = 1
print(ship)