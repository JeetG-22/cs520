import ship
import baseline_bot
import own_bot
import own_bot2


N = 30
num_ships = 20

base_count = 0
own_count = 0
own2_count = 0

ship_list = []
for _ in range(0, num_ships):
    # Dimensions set to 30x30
    spaceship = ship.Ship(N)

    # Place bot and rat
    spaceship.place_entities()
    ship_list.append(spaceship)
    # print(spaceship.grid)
    


for thresh in [1 + i * .1 for i in range(0, 10)]:
    base_avg_moves = 0
    own_avg_moves = 0
    own2_avg_moves = 0    
    for s_ship in ship_list:
        bot_base = baseline_bot.Baseline(s_ship)
        moves, ping_use, bot_base_pos = bot_base.find_rat(bot_base.get_est_pos(bot_base.get_position(2)), .1)
        base_avg_moves += moves

        bot_own = own_bot.Baseline(s_ship)
        moves_own, ping_use_own, bot_own_pos = bot_own.find_rat(bot_own.get_est_pos(bot_own.get_position(2)), .1, thresh)
        own_avg_moves += moves_own

        bot_own2 = own_bot2.Baseline(s_ship)
        moves_own2, ping_use_own2, bot_own_pos2 = bot_own2.find_rat(bot_own2.get_est_pos(bot_own2.get_position(2)), .1, thresh)
        own2_avg_moves += moves_own2
        if moves < moves_own and moves < moves_own2:
            base_count += 1
        elif moves_own < moves and moves_own < moves_own2:
            own_count += 1
        else:
            own2_count += 1
    print(f"({thresh}) - Baseline Bot Average Moves: {base_avg_moves / num_ships}")
    print(f"({thresh}) - Own Bot Average Moves: {own_avg_moves / num_ships}")
    print(f"({thresh}) - Own Bot 2 Average Moves: {own2_avg_moves / num_ships}")
    
print("Base Bot Win Count: " + str(base_count))
print("Own Bot Win Count: " + str(own_count))
print("Own 2 Bot Win Count: " + str(own2_count))
    

# print("Actual Bot Position: " + str(bot_base.get_position(2)))
# print("Estimated Bot Position: " + str(bot_base.get_est_pos(bot_base.get_position(2))))

# print("Baseline Bot # Of Moves: " + str(moves) + " || # Of Ping Usages: " + str(ping_use))
# print("Own Bot # Of Moves: " + str(moves_own) + " || # Of Ping Usages: " + str(ping_use_own))
# print("Own Bot 2 # Of Moves: " + str(moves_own2) + " || # Of Ping Usages: " + str(ping_use_own2))

# print("Ending Baseline Bot Position: " + bot_base_pos)
# print("Ending Own Bot Position: " + bot_own_pos)
# print("Ending Own Bot 2 Position: " + bot_own_pos2)

# print("Rat Actual Position: "+ str(bot_own.get_position(3)))
