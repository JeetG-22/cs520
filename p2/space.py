import ship
import baseline_bot
import own_bot
import matplotlib.pyplot as plt
import numpy as np

# N = 30
# num_of_sims = 100

# base_count = 0
# own_count = 0

# for _ in range(0, num_of_sims):
#     # Dimensions set to 30x30
#     spaceship = ship.Ship(N)

#     # Place bot and rat
#     spaceship.place_entities()
    

#     bot_base = baseline_bot.Baseline(spaceship)
#     moves, ping_use, bot_base_pos = bot_base.find_rat(bot_base.get_est_pos(bot_base.get_position(2)), .1)

#     bot_own = own_bot.Baseline(spaceship)
#     moves_own, ping_use_own, bot_own_pos = bot_own.find_rat(bot_own.get_est_pos(bot_own.get_position(2)), .1)
    
#     if moves < moves_own:
#         base_count += 1
#     else:
#         own_count += 1
    
# print("Base Bot Win Count: " + str(base_count))
# print("Own Bot Win Count: " + str(own_count))
    

# print("Actual Bot Position: " + str(bot_base.get_position(2)))
# print("Estimated Bot Position: " + str(bot_base.get_est_pos(bot_base.get_position(2))))

# print("Baseline Bot # Of Moves: " + str(moves) + " || # Of Ping Usages: " + str(ping_use))
# print("Own Bot # Of Moves: " + str(moves_own) + " || # Of Ping Usages: " + str(ping_use_own))
# print("Own Bot 2 # Of Moves: " + str(moves_own2) + " || # Of Ping Usages: " + str(ping_use_own2))

# print("Ending Baseline Bot Position: " + bot_base_pos)
# print("Ending Own Bot Position: " + bot_own_pos)
# print("Ending Own Bot 2 Position: " + bot_own_pos2)

# print("Rat Actual Position: "+ str(bot_own.get_position(3)))

# Evaluate your bot vs. the baseline bot, reporting a comparison of performance.
# Plot your results as a function of alpha.

# 100 simulations with each alpha
alpha = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

N = 30
num_of_sims = 100

base_moves = []
own_moves = []

for a in alpha:

    total_own_moves = 0
    total_base_moves = 0

    for _ in range(0, num_of_sims):
        # Dimensions set to 30x30
        spaceship = ship.Ship(N)

        # Place bot and rat
        spaceship.place_entities()

        bot_base = baseline_bot.Baseline(spaceship)
        moves, ping_use, bot_base_pos = bot_base.find_rat(bot_base.get_est_pos(bot_base.get_position(2)), a)

        bot_own = own_bot.Baseline(spaceship)
        moves_own, ping_use_own, bot_own_pos = bot_own.find_rat(bot_own.get_est_pos(bot_own.get_position(2)), a)

        total_base_moves += moves
        total_own_moves += moves_own

    base_moves.append(total_base_moves/num_of_sims)
    own_moves.append(total_own_moves/num_of_sims)

plt.plot(alpha, base_moves, label = "Baseline Bot")
plt.plot(alpha, own_moves, label = "Improved Bot")
plt.xlabel('Alpha Value')
plt.ylabel('Average Number of Moves to Catch Rat (100 trials)')
plt.title('Average Number of Moves Taken to Catch Rat vs. Alpha')
plt.legend()
plt.show()
plt.savefig("Plot2.png")