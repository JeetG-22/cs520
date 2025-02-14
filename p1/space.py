from ship import Ship
from bot1 import Bot1

vessel = Ship(D = 5)
vessel.place_entities()
print(vessel)

bot = Bot1(vessel)
print(bot.create_plan())