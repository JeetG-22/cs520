import ship
from heapq import heappop, heappush
from collections import deque
import sys

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
        queue = []
        est_cost = {}  # estimated total cost
        start_cost = {}  # stores the cost of each node from the beginning position
        
        # Initialize
        self.compute_fire_distances()
        start_cost[bot_start] = 0
        est_cost[bot_start] = self.heuristic(bot_start, button_start)
        heappush(queue, (est_cost[bot_start], bot_start))

        while queue:
            _, curr = heappop(queue)  # gets new bot position with minimum priority

            if self.SHIP.grid[curr[0]][curr[1]] == 4:  # reached button
                return True
            
            self.SHIP.spread_fire(flammability)

            if self.SHIP.grid[curr[0]][curr[1]] == 3:  # bot caught on fire
                return False
            
            self.compute_fire_distances()  # precompute each cell's distance to the fire
            
            # Get all neighbors
            for (dx, dy) in self.SHIP.neighbour_directions:
                neighbor = (curr[0] + dx, curr[1] + dy)

                # Evaluate only if it's not a fire or closed cell and it's in the grid
                if 0 <= neighbor[0] < self.SHIP.N and 0 <= neighbor[1] < self.SHIP.N and self.SHIP.grid[neighbor[0]][neighbor[1]] not in [0, 3]:

                    temp = start_cost[curr] + 1

                    if (neighbor not in start_cost or temp < start_cost[neighbor]):
                        start_cost[neighbor] = temp
                        est_cost[neighbor] = start_cost[neighbor] + self.heuristic(neighbor, button_start)

                        if neighbor not in queue:
                            heappush(queue, (est_cost[neighbor], neighbor))  # explore stronger options first
        
        return False
    

    # run each time spread_fire is called to track the distance for each cell to closest fire
    def compute_fire_distances(self):
        fire_dist = [[float('inf')] * self.SHIP.N for _ in range(self.SHIP.N)]
        queue = deque()

        # Get all fire cells
        for x in range(self.SHIP.N):
            for y in range(self.SHIP.N):

                if self.SHIP.grid[x][y] == 3:
                    queue.append((x, y, 0))  # (row, col, dist to nearest fire)
                    fire_dist[x][y] = 0

        # BFS for each cell
        while queue:
            x, y, dist = queue.popleft()

            # for all neighbors
            for dx, dy in self.SHIP.neighbour_directions:
                nx, ny = x + dx, y + dy

                # if it's an open cell
                if 0 <= nx < self.SHIP.N and 0 <= ny < self.SHIP.N and self.SHIP.grid[nx][ny] in [1, 2, 4]:
                    if dist + 1 < fire_dist[nx][ny]:
                        fire_dist[nx][ny] = dist + 1  # increment the fire distance
                        queue.append((nx, ny, dist + 1))

        self.fire_distance_map = fire_dist


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
        try:
            closest_fire_dist = self.fire_distance_map[bot_pos[0]][bot_pos[1]]
        except Exception as e:
            print(button_pos)
            print(self.fire_distance_map)
            print(self.SHIP.grid)
            sys.exit()
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