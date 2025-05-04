import ship
import bot
import random

# Note: ship does not have closed cells bordering its edges for some reason
s = ship.Ship(10)
s.place_entities()
print(s)

b = bot.Bot(s)
print(b.get_moves())

target = b.get_dead_end_cell()

# List of possible locations with bot pos in the beginning
L_values = [b.actual_bot_pos, target] + [cell for cell, _ in s.open_cells.items() if cell != b.actual_bot_pos and cell != target] 
print(L_values)

for i in range(1, len(L_values) + 1):
    L_list = L_values[:i]
    L_table = {k: k for k in L_list}
    print(L_table)
    pos, num_moves = b.get_moves(L_table)
    print(num_moves)

num_moves_per_L = []

# for _ in range(100):
#     s = ship.Ship(20)
#     s.place_entities()
#     b = bot.Bot(s)

#     # List of possible locations with bot pos in the beginning
#     L_values = [b.actual_bot_pos] + [cell for cell, _ in s.open_cells.items() if cell != b.actual_bot_pos] 

#     for i in range(1, len(L_values) + 1):
#         L_list = L_values[:i]
#         L_table = {k: k for k in L_list}
#         pos, num_moves = b.get_moves(L_table)
#         num_moves_per_L.append(num_moves)

# print(num_moves_per_L)