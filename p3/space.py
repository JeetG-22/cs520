import ship
import bot

# Note: ship does not have closed cells bordering its edges for some reason
s = ship.Ship(5)
s.place_entities()
print(s)

b = bot.Bot(s)
print(f"Actual pos: {b.get_position(2)}, Estimated pos: {b.get_est_pos()}")