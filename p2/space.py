import ship
import baseline_bot
import own_bot
N = 30

# Dimensions set to 30x30
spaceship = ship.Ship(N)

# Place bot and rat
spaceship.place_entities()

print(spaceship.grid)

bot_base = baseline_bot.Baseline(spaceship)
moves, ping_use, bot_base_pos = bot_base.find_rat(bot_base.get_est_pos(bot_base.get_position(2)), .1)

bot_own = own_bot.Baseline(spaceship)
moves_own, ping_use_own, bot_own_pos = bot_own.find_rat(bot_own.get_est_pos(bot_own.get_position(2)), .1)

print("Actual Bot Position: " + str(bot_base.get_position(2)))
print("Estimated Bot Position: " + str(bot_base.get_est_pos(bot_base.get_position(2))))

print("Baseline Bot # Of Moves: " + str(moves) + " || # Of Ping Usages: " + str(ping_use))
print("Own Bot # Of Moves: " + str(moves_own) + " || # Of Ping Usages: " + str(ping_use_own))

print("Ending Baseline Bot Position: " + bot_base_pos)
print("Ending Own Bot Position: " + bot_own_pos)

print("Rat Actual Position: "+ str(bot_own.get_position(3)))
