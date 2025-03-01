import ship
from queue import PriorityQueue

# Uses A* to find the shortest path to the button
# 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = fire cell, 4 = button cell
class Bot4:

    def __init__(self, SHIP):
        self.SHIP = SHIP

    # Returns true if the bot successfully gets to the button without
    # hitting a fire cell in its path.
    def mission_success(self, flammability):
        bot_pos = self.get_position()
        visited_positions = [bot_pos]
        
        #loop until bot finds correct path or fails
        while True:
            path = self.get_path(bot_pos, avoid_adj_fire=True)
            
            #check if there is a path with no open cell burning neighbors
            if path is None:
                path = self.get_path(bot_pos, avoid_adj_fire=False)
            
            #check if there is a path avoiding currently burning cells
            if path is None:
                return False, []
            
            # print(path)
        
            # move bot to the next cell & update the new position on grid for the next path 
            next_pos = path[1]
            bot_pos = next_pos
            
            #update new path 
            visited_positions.append(bot_pos)

            # if button cell is reached, return True
            if self.SHIP.grid[bot_pos[0]][bot_pos[1]] == 4:
                return True, visited_positions
        
            # spread the fire after bot moves
            self.SHIP.spread_fire(flammability)
           
            # check to see if the new fire spread is on the bot's current position
            if self.SHIP.grid[bot_pos[0]][bot_pos[1]] == 3:
                return False, []


    def get_path(self, curr_pos, avoid_adj_fire):

        # Get initial positions
        self.bot_start = curr_pos(2)
        button_start = curr_pos(4)
        fire_start = curr_pos(3)
        
        # Create structures
        queue = PriorityQueue()
        est_cost = {}  # estimated total cost
        start_cost = {}  # stores the cost of each node from the beginning position
        
        # Initialize
        start_cost[self.bot_start] = 0
        est_cost[self.bot_start] = heuristic(self.bot_start, button_start, fire_start)



    # Returns true if cell has neighbors that are burning
    def has_adj_fires(self, neighbor):

        for (dx, dy) in self.SHIP.neighbour_directions:
            adj_neighbor = (neighbor[0] + dx, neighbor[1] + dy)
            if 0 <= adj_neighbor[0] < self.SHIP.N and 0 <= adj_neighbor[1] < self.SHIP.N:
                if self.SHIP.grid[adj_neighbor[0]][adj_neighbor[1]] == 3:
                    return True
                
        return False
    
    # Returns coordinates of the indicated value
    def get_position(self, target):
        pos = (0,0)

        # Find initial position of target
        for i in range(self.SHIP.N):
            for j in range(self.SHIP.N):
                if self.SHIP.grid[i][j] == target:
                    pos = (i,j)
                    return pos