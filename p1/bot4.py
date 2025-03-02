from queue import PriorityQueue
from collections import deque

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
            path = self.find_path(bot_pos, button_pos, avoid_adjacent_fire=True)

            if not path:  # If no safe path exists, try again without avoiding adjacent fire
                path = self.find_path(bot_pos, button_pos, avoid_adjacent_fire=False)
                if not path:
                    return False  # No possible path to button

            bot_pos = path[1]  # Move to the next step in the path

            # Check if we reached the button
            if bot_pos == button_pos:
                return True
            
            # Spread the fire
            self.SHIP.spread_fire(flammability)

            # If bot steps into fire, fail
            if self.SHIP.grid[bot_pos[0]][bot_pos[1]] == 3:
                return False

    def find_path(self, start, goal, avoid_adjacent_fire=True):
        """Uses A* to find the shortest path while avoiding fire and optionally adjacent fire cells."""
        queue = PriorityQueue()
        queue.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}
        visited = set()

        while not queue.empty():
            _, curr = queue.get()
            if curr in visited:
                continue
            visited.add(curr)

            if curr == goal:
                return self.reconstruct_path(came_from, curr)

            for dx, dy in self.SHIP.neighbour_directions:
                neighbor = (curr[0] + dx, curr[1] + dy)

                if not (0 <= neighbor[0] < self.SHIP.N and 0 <= neighbor[1] < self.SHIP.N):
                    continue  # Out of bounds
                
                if self.SHIP.grid[neighbor[0]][neighbor[1]] == 0:  # Wall
                    continue
                
                if self.SHIP.grid[neighbor[0]][neighbor[1]] == 3:  # Fire
                    continue
                
                if avoid_adjacent_fire and self.is_adjacent_to_fire(neighbor):
                    continue  # Avoid adjacent fire cells if possible

                tentative_g_score = g_score[curr] + 1
                if tentative_g_score < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = curr
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                    queue.put((f_score[neighbor], neighbor))

        return None  # No path found

    # Documents path of the algorithm
    def get_path(self, came_from, curr):
        path = [curr]
        while curr in came_from:
            curr = came_from[curr]
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