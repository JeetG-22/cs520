from ship import Ship
from bot1 import Bot1
from bot2 import Bot2
from bot3 import Bot3
from bot4 import Bot4
import matplotlib.pyplot as plt
import copy
import numpy as np
from test_winnability import TestWinnability

# N = 10 
q_values = [round(i * .05, 2) for i in range (1, 21)]
# bot1_q_successes = []
# bot2_q_successes = []
# bot3_q_successes = []
# bot4_q_successes = []

# count_suc1 = count_fail1 = count_suc2 = count_fail2 = count_suc3 = count_fail3 = count_suc4 = count_fail4 = 0

# # Question 3
# for q in q_values:  # for each q
#     count_suc1 = 0
#     count_suc2 = 0
#     count_suc3 = 0
#     count_suc4 = 0

#     for i in range(0, 100):  # for 100 trials
#         vessel = Ship(D = 40)

#         bot1 = Bot1(copy.deepcopy(vessel))
#         if bot1.mission_success(1)[0]:
#             count_suc1 += 1
#         else:
#             count_fail1 += 1
            
#         bot2 = Bot2(copy.deepcopy(vessel))
#         if bot2.mission_success(1)[0]:
#             count_suc2 += 1
#         else:
#             count_fail2 += 1
            
#         bot3 = Bot3(copy.deepcopy(vessel))
#         if bot3.mission_success(1)[0]:
#             count_suc3 += 1
#         else:
#             count_fail3 += 1
        
#         bot4 = Bot4(copy.deepcopy(vessel))
#         if bot4.mission_success(1, 1):
#             count_suc4 += 1
#         else:
#             count_fail4 += 1

#     bot1_q_successes.append(count_suc1)
#     bot2_q_successes.append(count_suc2)
#     bot3_q_successes.append(count_suc3)
#     bot4_q_successes.append(count_suc4)

# # Create plots
# plt.figure(figsize = (10, 6))

# # Plot success rates for each bot with different colors and markers
# plt.plot(q_values, bot1_q_successes, label = "Bot1")
# plt.plot(q_values, bot2_q_successes, label = "Bot2")
# plt.plot(q_values, bot3_q_successes, label = "Bot3")
# plt.plot(q_values, bot4_q_successes, label = "Bot4")

# # Set y-axis limit
# plt.ylim(0, 100)

# # Labels and title
# plt.xlabel('q values')
# plt.ylabel('Successes out of 100 Trials')
# plt.title('Success Rates of Bots')

# # Add legend
# plt.legend()

# # Show plot
# plt.show()


# Question 4
# Run simulations for different values of q
num_trials = 100  # Number of simulations per q
winnable_frequencies = []
vessel = Ship(D = 40)

for q in q_values:
    winnable_count = 0
    for _ in range(num_trials):
        sim = TestWinnability(copy.deepcopy(vessel), q)
        if sim.is_winnable():
            winnable_count += 1
    winnable_frequencies.append(winnable_count / num_trials)

# Plot results
plt.figure(figsize=(8, 5))
plt.plot(q_values, winnable_frequencies, marker='o', linestyle='-')
plt.xlabel("Flammability (q)")
plt.ylabel("Probability of Winnability")
plt.title("Winnability of Simulation as a Function of Fire Spread Probability")
plt.grid()
plt.show()


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