from queue import PriorityQueue
from collections import deque

# 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = fire cell, 4 = button cell
class Bot4:

    def __init__(self, SHIP):
        self.SHIP = SHIP
        self.fire_distance_map = []
        
    # Returns true if the bot successfully gets to the button without
    # hitting a fire cell in its path.
    def mission_success(self, flammability, factor):

        # Get initial positions of the bot and button on the grid
        bot_pos = self.get_position(2)
        button_pos = self.get_position(4)

        # While the bot does not catch on fire
        while self.SHIP.grid[bot_pos[0]][bot_pos[1]] != 3:

            # Updates the fire_distance_map so that each cell on the map stores
            # how close that corresponding cell is to a fire cell on self.SHIP.grid
            self.compute_fire_distances()

            # Find shortest path while avoiding fire and adjacent fire cells if possible
            path = self.find_path(bot_pos, button_pos, factor, avoid_adjacent_fire = True)

            if not path:  # No safe path exists, try again without avoiding adjacent fire
                path = self.find_path(bot_pos, button_pos, factor, avoid_adjacent_fire = False)

                if not path:
                    return False 
                
            bot_pos = path[1]

            if bot_pos == button_pos:  # Reached button
                return True
            
            self.SHIP.spread_fire(flammability)

        return False
    
    # Uses A* to find the shortest path while avoiding fire and adjacent fire cells
    def find_path(self, start, goal, factor, avoid_adjacent_fire = True):

        # Initialize structs
        queue = PriorityQueue()
        # Keep track of the parent cell to return the path
        parent = {}
        # Keep track of the cost from the start
        start_cost = {}
        # Keep track of the estimated total cost
        est_cost = {}

        # Add start
        queue.put((0, start))
        start_cost[start] = 0
        est_cost[start] = self.heuristic(start, goal, factor)
        visited = set()

        # Begin A priori
        while queue.qsize() > 0:
            _, curr = queue.get()

            if curr not in visited:
                visited.add(curr)

                if curr == goal:  # Reached button coordinate
                    return self.get_path(parent, curr)

                # For all neighbors
                for dx, dy in self.SHIP.neighbour_directions:
                    neighbor = (curr[0] + dx, curr[1] + dy)

                    # If it's in the grid and not a closed or fire cell
                    if (0 <= neighbor[0] < self.SHIP.N and 0 <= neighbor[1] < self.SHIP.N) and self.SHIP.grid[neighbor[0]][neighbor[1]] not in [0, 3]:
                        
                        if avoid_adjacent_fire and self.is_adjacent_to_fire(neighbor):
                            continue
                        
                        # Evaluate the risk of that cell
                        temp = start_cost[curr]
                        fire_dist = self.fire_distance_map[neighbor[0]][neighbor[1]]
                        if fire_dist < 3:
                            temp += 3 - fire_dist

                        # Check if it's the best possible path
                        if neighbor not in start_cost or temp < start_cost[neighbor]:
                            parent[neighbor] = curr
                            start_cost[neighbor] = temp
                            est_cost[neighbor] = temp + self.heuristic(neighbor, goal, factor)
                            queue.put((est_cost[neighbor], neighbor))

        return None  # No path found

    # Documents path of the algorithm
    def get_path(self, parent_map, curr):
        path = [curr]

        while curr in parent_map:
            curr = parent_map[curr]
            path.append(curr)

        path.reverse()
        return path
    

    # Heuristic is defined by the proximity to fire and the button -- lower score is better
    def heuristic(self, bot_pos, button_pos, factor):  # ignore factor -- used for testing
        button_dist = abs(bot_pos[0] - button_pos[0]) + abs(bot_pos[1] - button_pos[1])
        closest_fire_dist = self.fire_distance_map[bot_pos[0]][bot_pos[1]]
        return button_dist - closest_fire_dist

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

    # Checks if a cell is right next to a fire cell, returns True if it is
    def is_adjacent_to_fire(self, cell):

        for dx, dy in self.SHIP.neighbour_directions:  # For each neighbor
            n = (cell[0] + dx, cell[1] + dy)

            if 0 <= n[0] < self.SHIP.N and 0 <= n[1] < self.SHIP.N:  # Inside the ship grid
                if self.SHIP.grid[n[0]][n[1]] == 3:  # Is it on fire?
                    return True
                
        return False


    # Returns coordinates of the given target value
    def get_position(self, target):
        for i in range(self.SHIP.N):
            for j in range(self.SHIP.N):
                if self.SHIP.grid[i][j] == target:
                    return (i, j)