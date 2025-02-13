from ship import Ship

# Uses BFS to find the shortest path to the button
# Avoids the initial fire cell but ignore the updated fire
class Bot1:

    def __init__(self, SHIP):
        self.SHIP = SHIP

    def create_plan(self):
        # Get source node
        self.bot_start = self.get_position()
        
        # Add it to the queue
        self.queue = []
        self.queue.append(self.bot_start)

        # Visited set
        visited = set()
        visited.add(self.bot_start)

        while self.queue:
            cell = self.queue.pop(0)
            print(cell)  # for debugging

            # Check if we reached the button
            if self.SHIP.grid[cell[0]][cell[1]] == 4:
                break

            # Get all neighbors
            for (dx, dy) in self.SHIP.neighbour_directions:
                
                pass
                # Check that it's not in the visited set

                # Check that its in the grid

                # Check that it's not a a fire cell

                # Add it to the queue

        
    def get_position(self):
        pos = (0, 0)

        # Find current position
        for i in range(self.SHIP.N):
            for j in range(self.SHIP.N):
                if self.SHIP.grid[i][j] == 2:
                    pos = (i, j)
                    break
        
        return pos