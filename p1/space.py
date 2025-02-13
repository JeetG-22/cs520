from ship import Ship
from bot1 import Bot1

vessel = Ship(D = 5)
vessel.place_entities()
print(vessel)
vessel.spread_fire(0.5)
print(vessel)
vessel.spread_fire(0.5)
print(vessel)
vessel.spread_fire(0.5)
print(vessel)

bot = Bot1(vessel)
bot.create_plan()