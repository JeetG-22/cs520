from ship import Ship
from bot1 import Bot1
from bot2 import Bot2
from bot3 import Bot3

# vessel = Ship(D = 10)
# vessel.place_entities()
# print("Original Vessel:\n\n")
# print(vessel)

# print("\nBot1")
# bot1 = Bot1(vessel)
# print(bot1.mission_success(.1))
# print(bot1.get_path())

# print("\n\nVessel After:")
# print(vessel)

# vessel2 = Ship(D = 10)
# vessel2.place_entities()

# print("\n\nOriginal Vessel:\n\n")
# print(vessel2)

# print("\nBot2")
# bot2 = Bot2(vessel2)
# print(bot2.mission_success(0.3))

# print("\n\nVessel After:")
# print(vessel2)

vessel3 = Ship(D = 7)
vessel3.place_entities()

print("\n\nOriginal Vessel:\n\n")
print(vessel3)

print("\nBot3")
bot3 = Bot3(vessel3)
print(bot3.mission_success(0.3))

print("\n\nVessel After:")
print(vessel3)


