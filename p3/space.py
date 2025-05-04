import ship
import bot
import math
import random
import matplotlib.pyplot as plt

# # Question 2 Plot
# num_moves_per_L = [0] * 10  # Each index will store the average moves at |L| = i/10th of the total open cells

# for i in range(100):
    
#     s = ship.Ship(10)
#     s.place_entities()
#     print(s)

#     b = bot.Bot(s)
#     print(b.get_moves())

#     target = b.get_dead_end_cell()

#     # List of possible locations with bot pos in the beginning
#     L_values = [b.actual_bot_pos, target] + [cell for cell, _ in s.open_cells.items() if cell != b.actual_bot_pos and cell != target] 
#     print(L_values)

#     for j in range(1, len(L_values) + 1):
#         L_list = L_values[:j]
#         random.shuffle(L_list)

#         L_table = {k: k for k in L_list}
#         pos, moves = b.get_moves(L_table)

#         index = math.floor((j - 1) / len(L_values) * 10)
#         num_moves_per_L[index] += moves

# avg_num_moves_per_L = [x / 100 for x in num_moves_per_L]
# L_values = [x / 10 for x in range(10)]

# plt.plot(L_values, avg_num_moves_per_L)
# plt.xlabel("Fraction of cells in L out of the total open cells")
# plt.ylabel("Average number of moves taken to localize")
# plt.title("Moves Needed to Localize Bot vs. |L|")
# plt.show()

# Plot 2
num_moves_per_L = [0] * 10 

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
        if len(L_list) > 2:
        # Get random subset of different initial L of that size
            L_list = [b.actual_bot_pos, target] + random.sample([cell for cell, _ in s.open_cells.items() if cell != b.actual_bot_pos and cell != target], len(L_list) - 2)
        
        random.shuffle(L_list)
        L_table = {k: k for k in L_list}
        pos, moves = b.get_moves(L_table)

        index = math.floor((j - 1) / len(L_values) * 10)
        num_moves_per_L[index] += moves

avg_num_moves_per_L = [x / 100 for x in num_moves_per_L]
L_values = [x / 10 for x in range(10)]

plt.plot(L_values, avg_num_moves_per_L)
plt.xlabel("Fraction of cells in L out of the total open cells")
plt.ylabel("Average number of moves taken to localize")
plt.title("Moves Needed to Localize Bot vs. |L|")
plt.show()