import numpy as np
import random

class Ship:
    # -1 = button cell, 0 = closed cell, 1 = open cell, 2 = fire cell
    SHIP = None
    N = None
    open_cells = {}
    def __init__(self, D):
        self.N = D
        self.SHIP = np.zeros((D, D), dtype=int)
        self.init_ship()
        
    def init_ship(self):
        list_one_neighbour = [] #for efficiency when randomly selecting a cell with one neighbour
        set_one_neighbour = set()
        
        initial_open_cell_row = random.randint(1, self.N-2)
        initial_open_cell_col = random.randint(1, self.N-2)
        self.SHIP[initial_open_cell_row][initial_open_cell_col] = 1 # Randomly place a open cell on the ship
        
        self.open_cells[(initial_open_cell_row, initial_open_cell_col)] = 0
        
        #add all the initial neighbours of the open cell 
        set_one_neighbour.add((initial_open_cell_row + 1, initial_open_cell_col))
        set_one_neighbour.add((initial_open_cell_row - 1, initial_open_cell_col))
        set_one_neighbour.add((initial_open_cell_row, initial_open_cell_col + 1))
        set_one_neighbour.add((initial_open_cell_row, initial_open_cell_col - 1))
        list_one_neighbour = list(set_one_neighbour)
        
        neighbour_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while(list_one_neighbour):
            #pick a random closed cell with one open neighbour
            random_cell = list_one_neighbour.pop(random.randint(0, len(list_one_neighbour) - 1))
            self.open_cells[random_cell] = 1
            self.SHIP[random_cell[0]][random_cell[1]] = 1 #update ship
            
            #add neighbours to set/list
            for(i, j) in neighbour_directions:
                #grid constraint edge cases
                if(random_cell[0] + i >= 0 and random_cell[0] + i < self.N and random_cell[1] + j >= 0 and random_cell[1] + j < self.N):
                    neighbour = (random_cell[0] + i, random_cell[1] + j)
                    if(neighbour in self.open_cells): #check to see if it is an open cell already
                        self.open_cells[neighbour] += 1
                        continue
                    
                    #check to see if adding this closed cell will potentially give it more than 1 open neighbour
                    if(neighbour in set_one_neighbour): 
                        if(neighbour in list_one_neighbour): #case for if it is deleted in a previous call
                            list_one_neighbour.remove(neighbour)
                    else:
                        set_one_neighbour.add(neighbour)
                        list_one_neighbour.append(neighbour)
        
        print(f"% Of Open Cells Before Deadend: {100 * np.count_nonzero(self.SHIP)/(self.N**2)}%")
        
        #init deadends
        deadend_open_cells = [k for k, v in self.open_cells.items() if v == 1]
        
        #for randomization of half the cells chosen
        DEADEND_RATE = len(deadend_open_cells) // 2 #~50%
        while(len(deadend_open_cells) != DEADEND_RATE):
            deadend_open_cells.pop(random.randint(0, len(deadend_open_cells) - 1))
        
        #convert to dictionary 
        deadend_open_cells = {item: [] for item in deadend_open_cells}
        
        #gives you all the closed cells that have at least 1 open neighbour 
        closed_cells = set_one_neighbour - self.open_cells.keys()
        
        for coords in deadend_open_cells.keys():
            row, col = coords
            #get all neighbours of deadend
            for(i, j) in neighbour_directions: #TODO: see if there is a more efficient solution
                potential_neightbor = (row + i, col + j)
                if(potential_neightbor in closed_cells): #no need to check grid constraints with the closed_cells set
                    deadend_open_cells[coords].append(potential_neightbor)
            
            #select random neighbour to be an open cell
            closed_random_neighbour = deadend_open_cells[coords].pop(random.randint(0, len(deadend_open_cells[coords]) - 1))
            self.SHIP[closed_random_neighbour[0]][closed_random_neighbour[1]] = 1
            self.open_cells[closed_random_neighbour] = 1 #TODO: not correct: figure out how to get the count of open neighbours efficiently
            self.open_cells[coords] += 1 #update open neighbour count
        print(f"% Of Open Cells After Deadend: {100 * np.count_nonzero(self.SHIP)/(self.N**2)}%")
                    
            
    def __str__(self):
        output = ''
        for i in range(self.N):
            for j in range(self.N):
                output += f'{self.SHIP[i][j]} '
            output += '\n'
        return f'Vessel:\n{output}Shape: {self.SHIP.shape}'
        
