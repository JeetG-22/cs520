from ship import Ship
from bot1 import Bot1

vessel = Ship(D = 40)
vessel.place_entities()
print(vessel)

bot = Bot1(vessel)
print(bot.mission_success(1))