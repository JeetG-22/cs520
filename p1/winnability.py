import copy

class Winnability:
    def __init__(self, original_ship, final_ship, max_timesteps, q):
         #used to get the original start position of the bot in case its already on fire in the final ship orientation
        self.final_ship = copy.deepcopy(final_ship)
        self.q = q
        self.N = final_ship.N
        self.timesteps_allowed = max_timesteps
        self.bot_start = self.find_position(original_ship.grid, 2)
    
        
    def is_winnable(self):
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
                if self.final_ship.grid[cell[0]][cell[1]] == 4:
                    return self.get_solution(parent, cell)

                # Get all neighbors
                for (dx, dy) in self.final_ship.neighbour_directions:
                    
                    neighbor = (cell[0] + dx, cell[1] + dy)

                    # Check that it's not in the visited set
                    if neighbor not in visited:

                        # Check that its in the grid
                        if 0 <= neighbor[0] < self.N and 0 <= neighbor[1] < self.N:

                            # Check that it's not a a fire cell or closed cell
                            if self.final_ship.grid[neighbor[0]][neighbor[1]] != 3 and self.final_ship.grid[neighbor[0]][neighbor[1]] != 0:

                                # Add it to the queue
                                self.queue.append(neighbor)
                                visited.add(neighbor)
                                parent[neighbor] = cell

            return []  # No solution
        
    # Returns the solution path once BFS finds the button
    def get_solution(self, parent, end):

        path = []

        while end is not None:
            path.append(end)
            end = parent[end]

        path.reverse()
        return path
        
    def find_position(self, grid, value):
        for i in range(self.N):
            for j in range(self.N):
                if grid[i][j] == value:
                    return (i, j)
        return None