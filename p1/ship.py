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
        print("Starting Cell: ", self.open_cells)
        
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
                        
        print("Set Dim: ", len(set_one_neighbour), "\n", set_one_neighbour)
        # print("List: ", list_one_neighbour)
        print("Open Cells Dim: ", len(self.open_cells), "\n", self.open_cells)
        
        print(np.count_nonzero(self.SHIP))
        print(np.coun)
        
        #init deadends
        OPEN_RATE = .5
        deadend_open_cells = {k: v for k, v in self.open_cells.items() if v == 1}
        print(deadend_open_cells)
        
        #gives you all the closed cells that have at least 1 open neighbour 
        closed_cells = set_one_neighbour - self.open_cells.keys()
        print(closed_cells)
        
        for key in deadend_open_cells.keys():
            row = key[0]
            col = key[1]
            
            
                
        
    
        
        
        
            
                
    def __str__(self):
        output = ''
        for i in range(self.N):
            for j in range(self.N):
                output += f'{self.SHIP[i][j]} '
            output += '\n'
        return f'Vessel:\n{output}Shape: {self.SHIP.shape}'
        
