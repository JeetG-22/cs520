import ship
import numpy as np

class Baseline:
    
    # 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = space rat cell
    def __init__(self, ship):
        self.spaceship = ship
        self.pos = self.get_position(2)
        self.rat_pos = self.get_position(3)
        self.eight_neighbor_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                                    (1, 1), (1, -1), (-1, 1), (-1, -1)]

        self.estimated_pos = ()

    # returns True if ping is heard
    # sens is a constant specifying the sensitivity of the detector
    def get_ping(self, sens):

        # get manhattan distance between rat and bot
        dist = abs(self.pos[0] - self.rat_pos[0]) + abs(self.pos[1] - self.rat_pos[1])

        prob = np.exp(-1 * sens * (dist - 1))

        if np.random.rand() < prob:
            return True
        
        return False


    # Returns the current position of the given val
    def get_position(self, val):
        pos = (0, 0)        
        # Find current position
        for i in range(self.spaceship.N):
            for j in range(self.spaceship.N):
                if self.spaceship.grid[i][j] == val:
                    pos = (i, j)
                    return pos


    # Sense how many of the neighboring eight cells are currently blocked
    def count_blocked_neighbors(self):
        count = 0
        for dx, dy in self.eight_neighbor_dirs:
            x = self.pos[0] + dx
            y = self.pos[1] + dy

            if 0 <= x <= 30 and 0 <= y <= 30 and self.spaceship[x][y] == 0:
                count += 1

        return count