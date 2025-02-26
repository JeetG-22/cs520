from ship import Ship
from bot1 import Bot1
from bot2 import Bot2

vessel = Ship(D = 10)
vessel.place_entities()
print("Original Vessel:\n\n")
print(vessel)

print("\nBot1")
bot1 = Bot1(vessel)
print(bot1.mission_success(.1))
print(bot1.get_path())

print("\n\nVessel After:")
print(vessel)

vessel2 = Ship(D = 10)
vessel2.place_entities()

print("\n\nOriginal Vessel:\n\n")
print(vessel2)

print("\nBot2")
bot2 = Bot2(vessel2)
print(bot2.mission_success(0.3))

print("\n\nVessel After:")
print(vessel2)


