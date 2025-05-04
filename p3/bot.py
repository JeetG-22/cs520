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
    

    # Implement the solution in part 2
    def get_moves(self):  # L is the set of possible locations
        
        # Get an open cell that is a dead end or corner
        # Get a list of dead end cells - cells that don't have neighbors on opposite sides
        dead_end_cells = self.L.copy()
        for cell in dead_end_cells.copy():
            x = cell[0]
            y = cell[1]
            if x - 1 >= 0 and x + 1 < self.spaceship.N and self.spaceship.grid[x-1][y] == 1 and self.spaceship.grid[x+1][y] == 1:
                dead_end_cells.pop(cell)
            elif y - 1 >= 0 and y + 1 < self.spaceship.N and self.spaceship.grid[x][y-1] == 1 and self.spaceship.grid[x][y+1] == 1:
                dead_end_cells.pop(cell)

        target, _ = random.choice(list(dead_end_cells.items()))

        # store the total sequence of moves
        all_moves = []

        while len(self.L) > 1:
            location, _ = random.choice(list(self.L.items()))

            # Get shortest path from location to target
            path = self.get_path(location, target)

            # Execute the sequence of moves and update L
            curr = location
            i = 1

            while i < len(path) and len(self.L) > 1:
                move = path[i]
                all_moves.append(move)
                dir = (move[0] - curr[0], move[1] - curr[1])
                self.L, success = self.get_L_next(self.L[self.actual_bot_pos], self.L, dir)
                curr = move
                i += 1
                if success != True:  # bot wasn't able to continue down this path
                    break
            
            # Reset L's values
            self.L = {k: k for k, _ in self.L.items()}
        
        return self.L.popitem()[0]


    def get_path(self, source, target):

        # Add it to the queue
        self.queue = []
        self.queue.append(source)

        # Visited set
        visited = set()
        visited.add(source)

        # Keep track of parent nodes/path
        parent = {source: None}

        while self.queue:
            cell = self.queue.pop(0)

            # Check if we reached the target
            if cell[0] == target[0] and cell[1] == target[1]:
                return self.get_solution(parent, cell)

            # Get all neighbors
            for (dx, dy) in self.spaceship.neighbour_directions:
                
                neighbor = (cell[0] + dx, cell[1] + dy)

                # Check that it's not in the visited set
                if neighbor not in visited:

                    # Check that its in the grid
                    if neighbor in self.spaceship.open_cells:

                            # Add it to the queue
                            self.queue.append(neighbor)
                            visited.add(neighbor)
                            parent[neighbor] = cell

        return None  # No solution
    

    # Returns the solution path once BFS finds the button
    def get_solution(self, parent, end):

        path = []

        while end is not None:
            path.append(end)
            end = parent[end]

        path.reverse()
        return path


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
        for pos, curr_pos in L.copy().items():
            pos_new = (curr_pos[0] + dir[0], curr_pos[1] + dir[1])
            if success:  # if the move worked and the new pos did not
                if pos_new not in self.spaceship.open_cells:
                    L.pop(pos)
                else:
                    L[pos] = pos_new
            else:
                # If the move failed, then positions that would have allowed the move should be removed
                if pos_new in self.spaceship.open_cells:
                    L.pop(pos)
    
        return L, success


    # Returns the current position of the given val
    def get_position(self, val):
        pos = (0, 0)        
        # Find current position
        for i in range(self.spaceship.N):
            for j in range(self.spaceship.N):
                if self.spaceship.grid[i][j] == val:
                    pos = (i, j)
                    return pos
                

    # Get bot's position (ignore this)
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
        
        return self.L.popitem()[0]