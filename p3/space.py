import ship
import bot
import math

# Question 2 Plot
num_moves_per_L = [0] * 10  # Each index will store the average moves at |L| = i/10th of the total open cells

for i in range(100):
    
    s = ship.Ship(10)
    s.place_entities()
    print(s)

    b = bot.Bot(s)
    print(b.get_moves())

    target = b.get_dead_end_cell()

    # List of possible locations with bot pos in the beginning
    L_values = [b.actual_bot_pos, target] + [cell for cell, _ in s.open_cells.items() if cell != b.actual_bot_pos and cell != target] 
    print(L_values)

    for j in range(1, len(L_values) + 1):
        L_list = L_values[:j]
        L_table = {k: k for k in L_list}
        pos, moves = b.get_moves(L_table)

        index = math.floor((j - 1) / len(L_values) * 10)
        num_moves_per_L[index] += moves
    
print(num_moves_per_L)