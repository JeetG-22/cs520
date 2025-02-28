import ship

# Uses BFS to find the shortest path to the button
# Avoids the initial fire cell but ignore the updated fire
# 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = fire cell, 4 = button cell
class Bot1:

    def __init__(self, SHIP):
        self.SHIP = SHIP

    # Returns true if the bot successfully gets to the button without
    # hitting a fire cell in its path.
    def mission_success(self, flammability):
        
        path = self.get_path()
        
        if path is None:
            return False, []

        curr = path[0]
        path_index = 0
        curr_val = self.SHIP.grid[curr[0]][curr[1]]

        # while the bot is not in a fire cell
        while curr_val != 3:

            # move bot to the next cell
            curr = path[path_index]
            curr_val = self.SHIP.grid[curr[0]][curr[1]]
            path_index += 1

            # if button cell is reached, return True
            if curr_val == 4:
                return True, path

            # spread the fire
            self.SHIP.spread_fire(flammability)
        
        # if it hits a fire cell, mission failed
        return False, []

    def get_path(self):
        # Get source node
        self.bot_start = self.get_position()
        
        # Add it to the queue
        self.queue = []
        self.queue.append(self.bot_start)

        # Visited set
        visited = set()
        visited.add(self.bot_start)

        # Keep track of parent nodes/path
        parent = {self.bot_start: None}

        while self.queue:
            cell = self.queue.pop(0)

            # Check if we reached the button
            if self.SHIP.grid[cell[0]][cell[1]] == 4:
                return self.get_solution(parent, cell)

            # Get all neighbors
            for (dx, dy) in self.SHIP.neighbour_directions:
                
                neighbor = (cell[0] + dx, cell[1] + dy)

                # Check that it's not in the visited set
                if neighbor not in visited:

                    # Check that its in the grid
                    if 0 <= neighbor[0] < self.SHIP.N and 0 <= neighbor[1] < self.SHIP.N:

                        # Check that it's not the initial fire cell or closed cell
                        if self.SHIP.grid[neighbor[0]][neighbor[1]] != 3 and self.SHIP.grid[neighbor[0]][neighbor[1]] != 0:

                            # Add it to the queue
                            self.queue.append(neighbor)
                            visited.add(neighbor)
                            parent[neighbor] = cell

        return None  # No solution
        

    def get_position(self):
        pos = (0, 0)        
        # Find current position
        for i in range(self.SHIP.N):
            for j in range(self.SHIP.N):
                if self.SHIP.grid[i][j] == 2:
                    pos = (i, j)
                    return pos

    
    # Returns the solution path once BFS finds the button
    def get_solution(self, parent, end):

        path = []

        while end is not None:
            path.append(end)
            end = parent[end]

        path.reverse()
        return path