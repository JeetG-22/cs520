import ship
import baseline_bot
import own_bot
import dynamic_bot
import baseline_move_bot
import own_move_bot
import matplotlib.pyplot as plt
import numpy as np

# Evaluate your bot vs. the baseline bot, reporting a comparison of performance.
# Plot your results as a function of alpha.

# 100 simulations with each alpha
alpha = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, .8, .9, 1]

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


#question 4
static_baseline_moves = []
static_own_moves = []
moving_baseline_moves = []
moving_improved_moves = []
moving_dynamic_moves = []  # For the specialized dynamic bot


for a in alpha:

    total_own_moves = 0
    total_base_moves = 0
    total_dyn_moves = 0

    for _ in range(0, num_of_sims):
        # Dimensions set to 30x30
        spaceship = ship.Ship(N)

        # Place bot and rat
        spaceship.place_entities()

        bot_base_move = baseline_move_bot.Baseline(spaceship)
        moves, ping_use, bot_base_pos = bot_base_move.find_rat(bot_base_move.get_est_pos(bot_base_move.get_position(2)), a)
        # print("Baseline Bot Finished")

        bot_own_move = own_move_bot.Baseline(spaceship)
        moves_own, ping_use_own, bot_own_pos = bot_own_move.find_rat(bot_own_move.get_est_pos(bot_own_move.get_position(2)), a)
        # print("Own Bot Finished")
        
        dyn_own_move = dynamic_bot.Baseline(spaceship)
        dyn_own_moves, ping_use_dyn, bot_dyn_pos = dyn_own_move.find_rat(dyn_own_move.get_est_pos(dyn_own_move.get_position(2)), a)
        # print("Dynamic Bot Finished")
        # print(a)
        
        

        total_base_moves += moves
        total_own_moves += moves_own
        total_dyn_moves += dyn_own_moves

    moving_baseline_moves.append(total_base_moves/num_of_sims)
    moving_improved_moves.append(total_own_moves/num_of_sims)
    moving_dynamic_moves.append(total_dyn_moves/num_of_sims)
    
    

plt.plot(alpha, moving_baseline_moves, label = "Baseline Bot")
plt.plot(alpha, moving_improved_moves, label = "Improved Bot")
plt.plot(alpha, moving_dynamic_moves, label = "Dynamic Bot")
plt.xlabel('Alpha Value')
plt.ylabel('Average Number of Moves to Catch Moving Rat (100 trials)')
plt.title('Average Number of Moves Taken to Catch Moving Rat vs. Alpha')
plt.legend()
plt.show()


# spaceship = ship.Ship(N)

# # Place bot and rat
# spaceship.place_entities()

# bot_base = baseline_move_bot.Baseline(spaceship)
# moves, ping_use, bot_base_pos = bot_base.find_rat(bot_base.get_est_pos(bot_base.get_position(2)), .1)
# print("Bot Position: " + str(bot_base_pos))
# print("Rat Position: " + str(bot_base.rat_pos))
# print("Base Bot Moves: " + str(moves))

# own_base = own_move_bot.Baseline(spaceship)
# moves_own, ping_use_own, bot_own_pos = own_base.find_rat(own_base.get_est_pos(own_base.get_position(2)), .1)
# print("Bot Position: " + str(bot_own_pos))
# print("Rat Position: " + str(own_base.rat_pos))
# print("Own Bot Moves: " + str(moves_own))

# dyn_bot = dynamic_bot.Baseline(spaceship)
# moves_dyn, ping_use_dyn, bot_dyn_pos = dyn_bot.find_rat(dyn_bot.get_est_pos(dyn_bot.get_position(2)), .1)
# print("Bot Position: " + str(bot_dyn_pos))
# print("Rat Position: " + str(dyn_bot.rat_pos))
# print("Dynamic Bot Moves: " + str(moves_dyn))


