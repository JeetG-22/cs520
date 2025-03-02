from queue import PriorityQueue

# 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = fire cell, 4 = button cell
class Bot4:

    def __init__(self, SHIP):
        self.SHIP = SHIP
        
    # Returns true if the bot successfully gets to the button without
    # hitting a fire cell in its path.
    def mission_success(self, flammability):
        bot_pos = self.get_position(2)
        button_pos = self.get_position(4)

        while True:
            # Find shortest path while avoiding fire and adjacent fire cells if possible
            path = self.find_path(bot_pos, button_pos, avoid_adjacent_fire = True)

            if not path:  # No safe path exists, try again without avoiding adjacent fire
                path = self.find_path(bot_pos, button_pos, avoid_adjacent_fire = False)
                
                if not path:
                    return False 
                
            bot_pos = path[1]

            if bot_pos == button_pos:  # Reached button
                return True
            
            self.SHIP.spread_fire(flammability)

            # If bot catches on fire
            if self.SHIP.grid[bot_pos[0]][bot_pos[1]] == 3:
                return False
    
    # Uses A* to find the shortest path while avoiding fire and adjacent fire cells
    def find_path(self, start, goal, avoid_adjacent_fire = True):

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
        est_cost[start] = self.heuristic(start, goal)
        visited = set()
        
        # Begin A priori
        while queue:
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
                        
                        # Increment the distance
                        temp = start_cost[curr] + 1

                        # Check if it's the best possible path
                        if neighbor not in start_cost or temp < start_cost[neighbor]:
                            parent[neighbor] = curr
                            start_cost[neighbor] = temp
                            est_cost[neighbor] = temp + self.heuristic(neighbor, goal)
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
    def heuristic(self, bot_pos, button_pos):
        button_dist = abs(bot_pos[0] - button_pos[0]) + abs(bot_pos[1] - button_pos[1])
        closest_fire_dist = self.closest_fire(bot_pos)
        return button_dist - 4/closest_fire_dist  # closeness to the button matters the most


    # Gets distance to the closest fire cell using BFS
    def closest_fire(self, cell):
        queue = []
        visited = set()
        visited.add(cell)

        while queue:
            (x, y), dist = queue.pop(0)
            
            # If it's on fire the distance is 0
            if self.SHIP.grid[x][y] == 3:
                return dist

            # For all neighbors
            for dx, dy in self.SHIP.neighbour_directions:
                neighbor = (x + dx, y + dy)

                # If it's inside the grid
                if 0 <= neighbor[0] < self.SHIP.N and 0 <= neighbor[1] < self.SHIP.N:

                    # If it hasn't been checked yet
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, dist + 1))  # Increment distance

        return float('inf')  # If fire isn't found -- this shouldn't be reached anyway


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