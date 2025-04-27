import ship
import numpy as np
import random

class Bot:
    
    # 0 = closed cell, 1 = open cell, 2 = bot cell
    def __init__(self, ship: ship.Ship):
        self.spaceship = ship
        self.actual_bot_pos = self.get_position(2)
        self.L = {k: k for k in self.spaceship.open_cells}  # hashtable storing the open cells initial position and current position

    # Attempt to move in a direction
    def get_new_pos(self, bot_pos, dir):
        new_pos_x = bot_pos[0] + dir[0]
        new_pos_y = bot_pos[1] + dir[1]

        if (new_pos_x, new_pos_y) in self.spaceship.open_cells:
            return (new_pos_x, new_pos_y)

        return bot_pos
    
    # Get bot's position
    def get_est_pos(self):
        curr_bot_pos = self.actual_bot_pos

        while len(self.L) > 1:

            # Attempt to move in all 4 directions
            for dir in self.spaceship.neighbour_directions:

                self.L, new_bot_pos = self.get_L_next(curr_bot_pos, self.L, dir)

                # If the bot was able to move, break the loop and try again
                if new_bot_pos != curr_bot_pos:                
                    curr_bot_pos = new_bot_pos
                    break
        
        return self.L.popitem()

    # Get the set of possible locations after moving
    def get_L_next(self, bot_pos, L: dict, dir):  # L = set of possible positions
    
        # Attempt to move
        new_bot_pos = self.get_new_pos(bot_pos, dir)

        # Check if bot was able to move
        if new_bot_pos == bot_pos:
            success = False
        else:
            success = True
        
        # Attempt move in other locations
        for pos in L.copy():
            pos_new = (pos[0] + dir[0], pos[1] + dir[1])
            if success:  # if the move worked and the new pos did not
                if pos_new not in self.spaceship.open_cells:
                    L.pop(pos)
                else:
                    L[pos] = pos_new
            else:
                # If the move failed, then positions that would have allowed the move should be removed
                if pos_new in self.spaceship.open_cells:
                    L.pop(pos)
    
        return L, new_bot_pos

    # Returns the current position of the given val
    def get_position(self, val):
        pos = (0, 0)        
        # Find current position
        for i in range(self.spaceship.N):
            for j in range(self.spaceship.N):
                if self.spaceship.grid[i][j] == val:
                    pos = (i, j)
                    return pos