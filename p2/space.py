import ship
import baseline_bot
import own_bot
import dynamic_bot


N = 30
num_of_sims = 100

base_count = 0
own_count = 0

# Dimensions set to 30x30
spaceship = ship.Ship(N)

# Place bot and rat
spaceship.place_entities()

bot_dynamic = dynamic_bot.Baseline(spaceship)
moves_dyn, ping_use_dyn, bot_dyn_pos = bot_dynamic.find_rat(bot_dynamic.get_est_pos(bot_dynamic.get_position(2)), .1)

print("Bot Final Position: " + str(bot_dyn_pos))
print("Rat Final Position: "+ str(bot_dynamic.rat_pos))

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
