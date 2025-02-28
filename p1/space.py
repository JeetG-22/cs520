from ship import Ship
from bot1 import Bot1
from bot2 import Bot2
from bot3 import Bot3
import copy

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

# vessel3 = Ship(D = 7)
# vessel3.place_entities()

# print("\n\nOriginal Vessel:\n\n")
# print(vessel3)

# print("\nBot3")
# bot3 = Bot3(vessel3)
# print(bot3.mission_success(0.3))

# print("\n\nVessel After:")
# print(vessel3)

N = 50

count_suc1 = 0
count_fail1 = 0
count_suc2 = 0
count_fail2 = 0
count_suc3 = 0
count_fail3 = 0

for i in range(0, N):
    vessel = Ship(D = 5)
    
    bot1 = Bot1(copy.deepcopy(vessel))
    if bot1.mission_success(1): 
        count_suc1 += 1
    else:
        count_fail1 += 1
        
    bot2 = Bot2(copy.deepcopy(vessel))
    if bot2.mission_success(1)[0]:
        count_suc2 += 1
    else:
        count_fail2 += 1
        
    bot3 = Bot3(copy.deepcopy(vessel))
    if bot3.mission_success(1)[0]:
        count_suc3 += 1
    else:
        count_fail3 += 1
    
print("Bot1: ") 
print("Successes: ", count_suc1)
print("Failures: ", count_fail1)

print("Bot2: ") 
print("Successes: ", count_suc2)
print("Failures: ", count_fail2)

print("Bot3: ") 
print("Successes: ", count_suc3)
print("FailureS: ", count_fail3)

# count_suc = count_fail = 0
# for i in range(0,N):
#     vessel = Ship(D = 40)
#     bot2 = Bot2(vessel)
#     if bot2.mission_success(.9)[0]:
#         count_suc += 1
#     else:
#         count_fail += 1
        
# print("Bot2: ") 
# print("Sucesses: ", count_suc)
# print("Failure", count_fail)

# count_suc = count_fail = 0
# for i in range(0,N):
#     vessel = Ship(D = 40)
#     bot3 = Bot3(vessel)
#     if bot3.mission_success(.9)[0]:
#         count_suc += 1
#     else:
#         count_fail += 1
        
# print("Bot3: ") 
# print("Sucesses: ", count_suc)
# print("Failure", count_fail)