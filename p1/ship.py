import numpy as np
import random

class Ship:

    # 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = fire cell, 4 = button cell

    # Creates ship with DxD blocked cells
    def __init__(self, D):
        self.N = D
        self.grid = np.zeros((D, D), dtype=int)
        self.open_cells = {}
        self.init_ship()
        self.neighbour_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.place_entities()
        
    # Opens cells to start creating the maze
    def init_ship(self):
        list_one_neighbour = []  # for efficiency when randomly selecting a cell with one neighbour
        set_one_neighbour = set()
        
        # Randomly open a square in the interior
        initial_open_cell_row = random.randint(1, self.N-2)
        initial_open_cell_col = random.randint(1, self.N-2)
        self.grid[initial_open_cell_row][initial_open_cell_col] = 1 
        
        self.open_cells[(initial_open_cell_row, initial_open_cell_col)] = 0
        
        # Add all the initial neighbours of the open cell 
        set_one_neighbour.add((initial_open_cell_row + 1, initial_open_cell_col))
        set_one_neighbour.add((initial_open_cell_row - 1, initial_open_cell_col))
        set_one_neighbour.add((initial_open_cell_row, initial_open_cell_col + 1))
        set_one_neighbour.add((initial_open_cell_row, initial_open_cell_col - 1))
        list_one_neighbour = list(set_one_neighbour)
        
        self.neighbour_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while(list_one_neighbour):
            # Pick a random closed cell with one open neighbour
            random_cell = list_one_neighbour.pop(random.randint(0, len(list_one_neighbour) - 1))
            self.open_cells[random_cell] = 1
            self.grid[random_cell[0]][random_cell[1]] = 1  # Update ship
            
            # Add neighbours to set/list
            for (i, j) in self.neighbour_directions:

                # If it's inside of the ship grid
                if (random_cell[0] + i >= 0 and random_cell[0] + i < self.N and random_cell[1] + j >= 0 and random_cell[1] + j < self.N):
                    neighbour = (random_cell[0] + i, random_cell[1] + j)
                    if (neighbour in self.open_cells):  # Check to see if it is an open cell already
                        self.open_cells[neighbour] += 1
                        continue
                    
                    # Check to see if adding this closed cell will potentially give it more than 1 open neighbour
                    if (neighbour in set_one_neighbour): 
                        if (neighbour in list_one_neighbour):  # case for if it is deleted in a previous call
                            list_one_neighbour.remove(neighbour)
                    else:
                        set_one_neighbour.add(neighbour)
                        list_one_neighbour.append(neighbour)
        
        # print(f"% Of Open Cells Before Deadend: {100 * np.count_nonzero(self.grid)/(self.N**2)}%")
        
        # Collect deadends
        deadend_open_cells = [k for k, v in self.open_cells.items() if v == 1]
        
        # Randomly select 50% of the deadends
        DEADEND_RATE = len(deadend_open_cells) // 2 #~50%
        random.shuffle(deadend_open_cells)
        deadend_open_cells = deadend_open_cells[0:DEADEND_RATE]
        
        # Convert to dictionary 
        deadend_open_cells = {item: [] for item in deadend_open_cells}
        # print(deadend_open_cells)
        
        # Gives you all the closed cells that have at least 1 open neighbour 
        closed_cells = set_one_neighbour - self.open_cells.keys()
        
        # For each dead-end in the list, open one closed neighbor
        for coords in deadend_open_cells.keys():
            row, col = coords
            # Get all neighbours of deadend
            for(i, j) in self.neighbour_directions: 
                potential_neightbor = (row + i, col + j)
                if(potential_neightbor in closed_cells): #no need to check grid constraints with the closed_cells set
                    deadend_open_cells[coords].append(potential_neightbor)
            
            # Select random neighbour to be an open cell
            closed_random_neighbour = deadend_open_cells[coords].pop(random.randint(0, len(deadend_open_cells[coords]) - 1))
            self.grid[closed_random_neighbour[0]][closed_random_neighbour[1]] = 1

            # TODO: Jeet - is this solution correct? Dynamically updates the count of open neighbors
            # print(closed_random_neighbour)
            sum = 0
            for (i, j) in self.neighbour_directions: 
                if ((closed_random_neighbour[0] + i, closed_random_neighbour[1] + j) in self.open_cells):
                    self.open_cells[(closed_random_neighbour[0] + i, closed_random_neighbour[1] + j)] += 1 #updates surrounding open neighbour counts
                    sum += 1
            self.open_cells[closed_random_neighbour] = sum
        
        # Double check if implemented correctly
        # print(f"% Of Open Cells After Deadend: {100 * np.count_nonzero(self.grid)/(self.N**2)}%")
        # print(self.open_cells)

    def place_entities(self):
        open_cells_list = list(self.open_cells.keys())
        fire_cell = open_cells_list.pop(random.randint(0, len(open_cells_list) - 1))
        self.grid[fire_cell[0]][fire_cell[1]] = 3
        button_cell = open_cells_list.pop(random.randint(0, len(open_cells_list) - 1))
        self.grid[button_cell[0]][button_cell[1]] = 4
        bot_cell = open_cells_list.pop(random.randint(0, len(open_cells_list) - 1))
        self.grid[bot_cell[0]][bot_cell[1]] = 2
        
        #pop these cells as they are not considered open anymore 
        self.open_cells.pop(fire_cell)
        self.open_cells.pop(button_cell)
        self.open_cells.pop(bot_cell)
        
        return fire_cell, button_cell, bot_cell

    # Executes one step of fire spreading with flammability q       
    def spread_fire(self, q):

        # Generate copy of the current ship
        copy = self.grid.copy()

        # For each cell check if its open and not burning
        # Count number of on fire neighbors
        for i in range(self.N):
            for j in range(self.N):

                if self.grid[i][j] not in [0, 3]:  # if it's a nonburning cell
                    count = 0   # count the number of burning cells

                    for (dx, dy) in self.neighbour_directions:
                        if (0 <= i + dx < self.N) and (0 <= j + dy < self.N):
                            count = count + 1 if self.grid[i + dx][j + dy] == 3 else count

                    if count > 0:  # won't spread if there's no burning neighbors
                        prob = 1 - (1 - q)**count
                        if random.random() < prob:
                            copy[i][j] = 3  # set on fire
        
        self.grid = copy
                
    def __str__(self):
        output = ''
        for i in range(self.N):
            for j in range(self.N):
                output += f'{self.grid[i][j]} '
            output += '\n'
        return f'Vessel:\n{output}Shape: {self.grid.shape}'