import ship
import bot

# Note: ship does not have closed cells bordering its edges for some reason
s = ship.Ship(5)
s.place_entities()
print(s)

b = bot.Bot(s)
print({b.get_moves(b.spaceship.open_cells)})