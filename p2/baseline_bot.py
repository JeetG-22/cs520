import ship
import numpy as np
import random

class Baseline:
    
    # 0 = closed cell, 1 = open cell, 2 = bot cell, 3 = space rat cell
    def __init__(self, ship):
        self.spaceship = ship
        self.pos = self.get_position(2)
        self.rat_pos = self.get_position(3)
        self.eight_neighbor_dirs = [(1, 0), (-1, 0), (0, 1), (0, -1),
                                    (1, 1), (1, -1), (-1, 1), (-1, -1)]

        self.estimated_pos = ()

    # Identify where the bot is
    def get_est_pos(self, actual_bot_pos):
        self.possible_positions = {k: k for k in self.spaceship.open_cells}  # the key will be the original position and the value is the currently tested position
        # print(self.possible_positions)

        # Repeat until we get the bot's true position
        curr = actual_bot_pos

        # Keep track of most open directions, so we don't get into a loop. Make sure the last four moves aren't the same. last move isnt the same as the third-last one, second to last move isnt the same as the current one
        most_open_dir_history = []

        loop = False

        while len(self.possible_positions) > 1:
            num_neighbors = self.count_blocked_neighbors(curr) 

            # Keep a knowledge base of open cells on the map
            for key, pos in self.possible_positions.copy().items():

                if self.count_blocked_neighbors(pos) != num_neighbors:  # if the bot's blocked cells don't match
                    self.possible_positions.pop(key)

            dir_frequency = {d: 0 for d in self.spaceship.neighbour_directions}

            for pos, curr_pos in self.possible_positions.items():

                for d in self.spaceship.neighbour_directions:
                    new_pos = (curr_pos[0] + d[0], curr_pos[1] + d[1])

                    if new_pos in self.spaceship.open_cells:
                        dir_frequency[d] += 1
        
            # Choose the direction with the max count
            self.new_dir = max(dir_frequency, key = dir_frequency.get)

            # check for loop:
            if len(most_open_dir_history) > 4 and (most_open_dir_history[-1] == most_open_dir_history[-3] and most_open_dir_history[-2] == self.new_dir):

                # Loop detected
                loop = True

            # Move in random direction each time to escape dead end.
            if loop:
                self.new_dir = random.choice(self.spaceship.neighbour_directions)

            most_open_dir_history.append(self.new_dir)

            # Attempt to move
            new_bot_pos = (curr[0] + self.new_dir[0], curr[1] + self.new_dir[1])
        
            if new_bot_pos in self.spaceship.open_cells:
                success = True
                curr = new_bot_pos
            else:
                success = False
            
            # Update the other positions and attempt move
            for pos, curr_pos in self.possible_positions.copy().items():
                pos_new = (curr_pos[0] + self.new_dir[0], curr_pos[1] + self.new_dir[1])  # current position
                if success:  # if the move worked and the new pos did not
                    if pos_new not in self.spaceship.open_cells:
                        self.possible_positions.pop(pos)
                    else:
                        self.possible_positions[pos] = pos_new
                else:
                    # If the move failed, then positions that would have allowed the move should be removed
                    if pos_new in self.spaceship.open_cells:
                        self.possible_positions.pop(pos)
        
        # Once only one candidate remains, set it as the estimated position.
        self.estimated_pos = self.possible_positions.popitem()
        return self.estimated_pos[0]
    
    def find_rat(self, est_pos, alpha):
        moves = ping_use = 0
        bot_pos = est_pos
        rat_kb = {} #knowledge base for rat (stores probabilities of open cells)
        
        #initialize all open cells probabilities to some uniform value
        count_open_cells = len(self.spaceship.open_cells)
        for cell_pos in self.spaceship.open_cells:
            if(cell_pos != bot_pos): #to not include bot cell
                rat_kb[cell_pos] = 1.0 / (count_open_cells - 1)
        # print("Rat Knowledge Base Original: ", str(rat_kb))
        
        current_path = []
                
        while True:
            ping_use += 1
            ping_found = self.get_ping(alpha, bot_pos)
            
            if(self.rat_detected(bot_pos)):
                # print("Rat Found!")
                break
            
            sum_prob = 0 #factor to make sure the probabilities add up to 1
            updated_rat_kb = {} #temporary storage instead of manipulating rat_kb directly during the loop
            for open_pos, prob in rat_kb.items():
                
                if(bot_pos == open_pos): #we know that can't be here so we don't add it to the rat KB. 
                    continue #removes the need for redistributing prob once we reach the target cell and find that the rat isn't there              
                
                # get manhattan distance between cell and bot estimated position
                dist = abs(open_pos[0] - bot_pos[0]) + abs(open_pos[1] - bot_pos[1])
                updated_prob = 0
                #two situations: ping is heard or ping is not heard
                if(ping_found): #formula: P(rat in cell | ping found) = (P(ping found | rat in cell) * P(rat in cell)) / P(ping found)
                    prob_ping = np.exp(-alpha * (dist - 1))
                    updated_prob = prob_ping * prob
                else: #formula: P(rat in cell | ping not found) = (P(ping not found | rat in cell) * P(rat in cell)) / P(ping not found)
                    prob_ping = 1 - np.exp(-alpha * (dist - 1))
                    updated_prob = prob_ping * prob
                updated_rat_kb[open_pos] = updated_prob
                sum_prob += updated_prob
            
            #to finish formula (dividing P(ping found) or P(ping not found))
            if sum_prob > 0:
                for cell in updated_rat_kb:
                    updated_rat_kb[cell] /= sum_prob
                
            #update rat knowledge base
            rat_kb = updated_rat_kb
            # print("Rat Knowledge Base: ", str(rat_kb))
            
            if not current_path:
                if(rat_kb):
                    current_target_cell = max(rat_kb, key=rat_kb.get)
                    # print("Target Cell: " , str(current_target_cell))
                    current_path = self.find_path(bot_pos, current_target_cell)
                     #TODO: what should we do about the edge case where a particular target cell has the highest prob but is unreachable given the bots current position. 
                     # should i remove it from the rat_kb or should i keep it a let the ping keep going until a new target cell is found (would cause a discrepency
                     # in the moves and the ping usages).
                    # if(not current_path):
                    #     print(str(current_path) + "in here~~~~~~~~~~~~~~~~~~")
                else:
                    print("Rat Knowledge Base Is Empty!")
                    break
            if current_path:
                bot_pos = current_path.pop(0)
                moves += 1
                
                if(self.rat_detected(bot_pos)): #recheck to see if we are in rat cell
                    # print("Ending Baseline Bot Position: " + str(bot_pos))
                    # print("Rat Found!")
                    break
        return moves, ping_use, str(bot_pos)
    
    #bfs to find path to target cell
    def find_path(self, start, end):
        queue = []
        queue.append(start)
        visited = set([start])
        parent = {start: None}
        
        while queue:
            cell = queue.pop(0)
            
            #return solution path 
            if(cell == end):
                path = []
                while(cell is not None):
                    path.append(cell)
                    cell = parent[cell]
                path.reverse()
                return path
            
            for cardinal_dir in self.spaceship.neighbour_directions:
                neighbour = (cardinal_dir[0] + cell[0], cardinal_dir[1] + cell[1])
                if(neighbour in self.spaceship.open_cells and neighbour not in visited):
                    queue.append(neighbour)
                    visited.add(neighbour)
                    parent[neighbour] = cell
        return []
                    
    # returns True if ping is heard
    # sens is a constant specifying the sensitivity of the detector
    def get_ping(self, sens, curr_bot_pos = None):
        
        if curr_bot_pos is None:
            curr_bot_pos = self.pos

        # get manhattan distance between rat and bot
        dist = abs(curr_bot_pos[0] - self.rat_pos[0]) + abs(curr_bot_pos[1] - self.rat_pos[1])

        prob = np.exp(-1 * sens * (dist - 1))

        if np.random.rand() < prob:
            return True
        
        return False

    #for definitive detection
    def rat_detected(self, bot_pos = None):
        if bot_pos is None:
            bot_pos = self.pos
        return bot_pos == self.rat_pos
        

    # Returns the current position of the given val
    def get_position(self, val):
        pos = (0, 0)        
        # Find current position
        for i in range(self.spaceship.N):
            for j in range(self.spaceship.N):
                if self.spaceship.grid[i][j] == val:
                    pos = (i, j)
                    return pos


    # Sense how many of the neighboring eight cells are currently blocked
    def count_blocked_neighbors(self, cell):
        count = 0
        for dx, dy in self.eight_neighbor_dirs:
            x = cell[0] + dx
            y = cell[1] + dy

            if 0 <= x <= self.spaceship.N and 0 <= y <= self.spaceship.N and self.spaceship.grid[x][y] == 0:
                count += 1

        return count