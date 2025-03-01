import ship
from queue import PriorityQueue

# Uses A* to find the shortest path to the button
# 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = fire cell, 4 = button cell
class Bot4:

    def __init__(self, SHIP):
        self.SHIP = SHIP

    # A Priori algorithm, returns true if bot successfully gets to button
    def mission_success(self, curr_pos, flammability):

        # Get initial positions
        bot_start = curr_pos(2)
        button_start = curr_pos(4)
        fire_start = curr_pos(3)
        
        # Create structures
        queue = PriorityQueue()
        est_cost = {}  # estimated total cost
        start_cost = {}  # stores the cost of each node from the beginning position
        
        # Initialize
        start_cost[self.bot_start] = 0
        est_cost[self.bot_start] = heuristic(bot_start, button_start)
        queue.put((est_cost[self.bot_start], bot_start))

        while queue:
            curr = queue.get()  # gets new bot position with minimum priority

            if self.SHIP.grid[curr[0]][curr[1]] == 4:  # reached button
                return True
            
            self.SHIP.spread_fire(flammability)

            if self.SHIP.grid[curr[0]][curr[1]] == 3:  # bot caught on fire
                return False
            
            # Get all neighbors
            for (dx, dy) in self.SHIP.neighbour_directions:
    
                neighbor = (curr[0] + dx, curr[1] + dy)

                temp = start_cost[curr] + 1

                if (neighbor in start_cost and temp < start_cost[neighbor]):
                    start_cost[neighbor] = temp
                    est_cost[neighbor] = start_cost[neighbor] + heuristic(neighbor, button_start)

                    if neighbor not in queue:
                        queue.put((est_cost[neighbor], neighbor))  # explore stronger options first
        
        return False
    
    def heuristic(self, bot_pos, button_pos):
        pass


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