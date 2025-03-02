from ship import Ship
from bot1 import Bot1
from bot2 import Bot2
from bot3 import Bot3
from bot4 import Bot4
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

N = 10 
q_values = [round(i * .05, 2) for i in range (1, 21)]
bot1_q_successes = []
bot2_q_successes = []
bot3_q_successes = []
bot4_q_successes = []

print(q_values)

count_suc1 = count_fail1 = count_suc2 = count_fail2 = count_suc3 = count_fail3 = count_suc4 = count_fail4 = 0

# Question 3
for q in q_values:  # for each q
    count_suc1 = 0
    count_suc2 = 0
    count_suc3 = 0
    count_suc4 = 0

    for i in range(0, 100):  # for 100 trials
        vessel = Ship(D = 15)

        bot1 = Bot1(copy.deepcopy(vessel))
        if bot1.mission_success(1)[0]:
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
        
        bot4 = Bot4(copy.deepcopy(vessel))
        if bot4.mission_success(1, 1):
            count_suc4 += 1
        else:
            count_fail4 += 1

    bot1_q_successes.append(count_suc1)
    bot2_q_successes.append(count_suc2)
    bot3_q_successes.append(count_suc3)
    bot4_q_successes.append(count_suc4)


print(bot1_q_successes)
print(bot2_q_successes)
print(bot3_q_successes)
print(bot4_q_successes)

# # Used to test best coefficients for heuristic
# for factor in range(1,12):
#     count_suc4 = 0
#     count_fail4 = 0
    
#     for i in range(0, 100):

#         vessel = Ship(D = 15)
#         bot4 = Bot4(copy.deepcopy(vessel))

#         if bot4.mission_success(1, factor):
#             count_suc4 += 1
#         else:
#             count_fail4 += 1

#     print(f"Bot4: {factor}") 
#     print("Successes: ", count_suc4)
#     print("Failures: ", count_fail4)