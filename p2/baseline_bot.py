import ship

class Baseline:

    def __init__(self, ship):
        self.spaceship = ship
        self.pos = self.get_position
        self.eight_neighbor_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                                    (1, 1), (1, -1), (-1, 1), (-1, -1)]

    

    # Returns the current position of the bot
    def get_position(self):
        pos = (0, 0)        
        # Find current position
        for i in range(self.spaceship.N):
            for j in range(self.spaceship.N):
                if self.spaceship.grid[i][j] == 2:
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