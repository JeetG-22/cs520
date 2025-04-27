import ship
import bot

# Note: ship does not have closed cells bordering its edges for some reason
s = ship.Ship(6)
s.place_entities()
print(s)

b = bot.Bot(s)
print(b.get_est_pos())