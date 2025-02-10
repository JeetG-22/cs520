import numpy as np
import random

class Ship:
    # -1 = button cell, 0 = closed cell, 1 = open cell, 2 = fire cell
    SHIP = None
    N = None
    open_cells = set()
    def __init__(self, D):
        self.N = D
        self.SHIP = np.zeros((D, D), dtype=int)
        self.init_ship()
        
    def init_ship(self):
        list_one_neighbour = [] #for efficiency when randomly selecting a cell with one neighbour
        dict_one_neighbour = {}
        
        initial_open_cell_row = random.randint(1, self.N-2)
        initial_open_cell_col = random.randint(1, self.N-2)
        self.SHIP[initial_open_cell_row][initial_open_cell_col] = 1 # Randomly place a open cell on the ship
        
        self.open_cells.add((initial_open_cell_row, initial_open_cell_col))
        
        #add all the initial neighbours of the open cell 
        dict_one_neighbour[(initial_open_cell_row + 1, initial_open_cell_col)] = 1
        dict_one_neighbour[(initial_open_cell_row - 1, initial_open_cell_col)] = 1
        dict_one_neighbour[(initial_open_cell_row, initial_open_cell_col + 1)] = 1
        dict_one_neighbour[(initial_open_cell_row, initial_open_cell_col - 1)] = 1
        list_one_neighbour = list(dict_one_neighbour)
        print(dict_one_neighbour)
        
        neighbour_directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        while(list_one_neighbour.__len__() > 0):
            #pick a random closed cell with one open neighbour
            random_cell = list_one_neighbour.pop(random.randint(0, len(list_one_neighbour) - 1))
            self.open_cells.add(random_cell)
            self.SHIP[random_cell[0]][random_cell[1]] = 1 #update ship
            
            #add neighbours to set/list
            for(i, j) in neighbour_directions:
                #grid constraint edge cases
                if(random_cell[0] + i >= 0 and random_cell[0] + i < self.N and random_cell[1] + j >= 0 and random_cell[1] + j < self.N):
                    neighbour = (random_cell[0] + i, random_cell[1] + j)
                    if(neighbour in self.open_cells): continue
                    #check to see if adding this closed cell will potentially give it more than 1 open neighbour
                    if(neighbour in dict_one_neighbour): 
                        dict_one_neighbour[neighbour] += 1
                        if(neighbour in list_one_neighbour): #case for if it is deleted in a previous call
                            list_one_neighbour.remove(neighbour)
                    else:
                        dict_one_neighbour[neighbour] = 1
                        list_one_neighbour.append(neighbour)
                        
            print(dict_one_neighbour)
            print(list_one_neighbour)
                
    def __str__(self):
        output = ''
        for i in range(self.N):
            for j in range(self.N):
                output += f'{self.SHIP[i][j]} '
            output += '\n'
        return f'Vessel:\n{output}Shape: {self.SHIP.shape}'
        
