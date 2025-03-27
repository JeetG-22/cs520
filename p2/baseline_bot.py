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

    # Identify where the bot is
    def get_est_pos(self, actual_bot_pos):
        self.possible_positions = {k: k for k in self.spaceship.open_cells}  # the key will be the original position and the value is the currently tested position

        # Repeat until we get the bot's true position
        curr = actual_bot_pos

        while len(self.possible_positions) > 1:
            num_neighbors = self.count_blocked_neighbors(curr) 

            # Keep a knowledge base of open cells on the map
            for key, pos in self.possible_positions.copy().items():

                if self.count_blocked_neighbors(pos) != num_neighbors:  # if the bot's blocked cells don't match
                    self.possible_positions.pop(key)

            dir_frequency = {d: 0 for d in self.spaceship.neighbour_directions}

            for pos, curr_pos in self.possible_positions.items():

                for d in self.spaceship.neighbour_directions:
                    new_pos = (curr_pos[0] + d[0], curr_pos[1] + d[1])

                    if new_pos in self.spaceship.open_cells:
                        dir_frequency[d] += 1
        
            # Choose the direction with the max count
            most_open_dir = max(dir_frequency, key = dir_frequency.get)
            # Attempt to move
            new_bot_pos = (curr[0] + most_open_dir[0], curr[1] + most_open_dir[1])
        
            if new_bot_pos in self.spaceship.open_cells:
                success = True
                curr = new_bot_pos
            else:
                success = False
            
            # Update the other positions and attempt move
            for pos, curr_pos in self.possible_positions.copy().items():
                pos_new = (curr_pos[0] + most_open_dir[0], curr_pos[1] + most_open_dir[1])  # current position
                if success:  # if the move worked and the new pos did not
                    if pos_new not in self.spaceship.open_cells:
                        self.possible_positions.pop(pos)
                    else:
                        self.possible_positions[pos] = pos_new
                else:
                    # If the move failed, then positions that would have allowed the move should be removed
                    if pos_new in self.spaceship.open_cells:
                        self.possible_positions.pop(pos)
        
        # Once only one candidate remains, set it as the estimated position.
        self.estimated_pos = self.possible_positions.popitem()
        return self.estimated_pos[0]

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
    def count_blocked_neighbors(self, cell):
        count = 0
        for dx, dy in self.eight_neighbor_dirs:
            x = cell[0] + dx
            y = cell[1] + dy

            if 0 <= x <= 30 and 0 <= y <= 30 and self.spaceship.grid[x][y] == 0:
                count += 1

        return count