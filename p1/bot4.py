import ship
from queue import PriorityQueue
from collections import deque

# Uses A* to find the shortest path to the button
# 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = fire cell, 4 = button cell
class Bot4:

    def __init__(self, SHIP):
        self.SHIP = SHIP

    # A Priori algorithm, returns true if bot successfully gets to button
    def mission_success(self, flammability):

        # Get initial positions
        bot_start = self.get_position(2)
        button_start = self.get_position(4)
        
        # Create structures
        queue = PriorityQueue()
        est_cost = {}  # estimated total cost
        start_cost = {}  # stores the cost of each node from the beginning position
        
        # Initialize
        start_cost[bot_start] = 0
        est_cost[bot_start] = self.heuristic(bot_start, button_start)
        queue.put((est_cost[bot_start], bot_start))

        while queue:
            _, curr = queue.get()  # gets new bot position with minimum priority

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
                    est_cost[neighbor] = start_cost[neighbor] + self.heuristic(neighbor, button_start)

                    if neighbor not in queue:
                        queue.put((est_cost[neighbor], neighbor))  # explore stronger options first
        
        return False
    

    # returns distance of the closest fire to the cell using BFS
    def closest_fire(self, cell):   

        # Stores the cell and its distance
        queue = deque([(cell, 0)])
        visited = set()
        visited.add(cell)

        while queue:
            (x, y), dist = queue.popleft()  # Pop the cell and its distance

            # Check if we reached the fire
            if self.SHIP.grid[x][y] == 3:
                return dist

            # Explore all neighbors
            for dx, dy in self.SHIP.neighbour_directions:
                neighbor = (x + dx, y + dy)

                # Check bounds
                if 0 <= neighbor[0] < self.SHIP.N and 0 <= neighbor[1] < self.SHIP.N:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))  # increase distance

        return float('inf')  # no fire?? shouldn't return this anyway


    #  Scores possible movement for the bot depending on fire positions and button position
    def heuristic(self, bot_pos, button_pos):
        button_dist = abs(bot_pos[0] - button_pos[0]) + abs(bot_pos[1] - button_pos[1])
        closest_fire_dist = self.closest_fire(bot_pos)
        
        return button_dist - closest_fire_dist
    

    # Returns coordinates of the indicated value
    def get_position(self, target):
        pos = (0,0)

        # Find initial position of target
        for i in range(self.SHIP.N):
            for j in range(self.SHIP.N):
                if self.SHIP.grid[i][j] == target:
                    pos = (i,j)
                    return pos