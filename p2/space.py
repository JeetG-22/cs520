import ship
import baseline_bot
N = 30

# Dimensions set to 30x30
spaceship = ship.Ship(N)

# Place bot and rat
spaceship.place_entities()

print(spaceship.grid)

base_bot = baseline_bot.Baseline(spaceship)
print("Actual Bot Position: " + str(base_bot.get_position(2)))
print("Estimated Bot Position: " + str(base_bot.get_est_pos(base_bot.get_position(2))))

moves, ping_use = base_bot.find_rat(base_bot.get_est_pos(base_bot.get_position(2)), .1)
print("Baseline Bot # Of Moves: " + str(moves) + " || # Of Ping Usages: " + str(ping_use))

own_bot = baseline_bot.Baseline(spaceship)

moves, ping_use = own_bot.find_rat(own_bot.get_est_pos(own_bot.get_position(2)), .1)
print("Own Bot # Of Moves: " + str(moves) + " || # Of Ping Usages: " + str(ping_use))
print("Rat Actual Position: "+ str(own_bot.get_position(3)))
