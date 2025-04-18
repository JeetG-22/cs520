import ship

# Uses BFS to find the shortest path to the button
# 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = fire cell, 4 = button cell
class Bot3:

    def __init__(self, SHIP):
        self.SHIP = SHIP

    # Returns true if the bot successfully gets to the button without
    # hitting a fire cell in its path.
    def mission_success(self, flammability):
       
        bot_pos = self.get_position()
        self.visited_positions = [bot_pos]
        
        #loop until bot finds correct path or fails
        while True:
            path = self.get_path(bot_pos, avoid_adj_fire=True)
            
            #check if there is a path with no open cell burning neighbors
            if path is None:
                path = self.get_path(bot_pos, avoid_adj_fire=False)
            
            #check if there is a path avoiding currently burning cells
            if path is None:
                return False, []
                    
            # move bot to the next cell & update the new position on grid for the next path 
            next_pos = path[1]
            bot_pos = next_pos
            
            #update new path 
            self.visited_positions.append(bot_pos)

            # if button cell is reached, return True
            if self.SHIP.grid[bot_pos[0]][bot_pos[1]] == 4:
                return True, self.visited_positions
        
            # spread the fire after bot moves
            self.SHIP.spread_fire(flammability)
           
            # check to see if the new fire spread is on the bot's current position
            if self.SHIP.grid[bot_pos[0]][bot_pos[1]] == 3:
                return False, self.visited_positions
            


    def get_path(self, curr_pos, avoid_adj_fire):

        # Get source node
        self.bot_start = curr_pos
        
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
                if neighbor in visited:
                    continue

                # Check that its in the grid
                if not (0 <= neighbor[0] < self.SHIP.N and 0 <= neighbor[1] < self.SHIP.N):
                    continue
            
                # Check that it's not a a fire cell or closed cell
                if self.SHIP.grid[neighbor[0]][neighbor[1]] != 3 and self.SHIP.grid[neighbor[0]][neighbor[1]] != 0:
                    
                    if(avoid_adj_fire and self.has_adj_fires(neighbor)):
                        continue

                    # Add it to the queue
                    self.queue.append(neighbor)
                    visited.add(neighbor)
                    parent[neighbor] = cell

        return None  # No solution
        

    #returns true if cell has neighbors that are burning
    def has_adj_fires(self, neighbor):
        for (dx, dy) in self.SHIP.neighbour_directions:
            adj_neighbor = (neighbor[0] + dx, neighbor[1] + dy)
            if 0 <= adj_neighbor[0] < self.SHIP.N and 0 <= adj_neighbor[1] < self.SHIP.N:
                if self.SHIP.grid[adj_neighbor[0]][adj_neighbor[1]] == 3:
                    return True
        return False
    
    def get_position(self):
        pos = (0,0)
        # Find initial position of bot
        for i in range(self.SHIP.N):
            for j in range(self.SHIP.N):
                if self.SHIP.grid[i][j] == 2:
                    pos = (i,j)
                    return pos
                
    
    def get_visited_positions(self):
        return self.visited_positions
    
    # Returns the solution path once BFS finds the button
    def get_solution(self, parent, end):

        path = []

        while end is not None:
            path.append(end)
            end = parent[end]

        path.reverse()
        return path