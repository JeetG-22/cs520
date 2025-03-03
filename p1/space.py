from ship import Ship
from bot1 import Bot1
from bot2 import Bot2
from bot3 import Bot3
from bot4 import Bot4
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import copy
import numpy as np
import random
from winnability import Winnability

N = 100 # number of trials
q_values = [round(i * .05, 2) for i in range (1, 21)]
bot1_q_successes = []
bot2_q_successes = []
bot3_q_successes = []
bot4_q_successes = []

count_suc1 = count_fail1 = count_suc2 = count_fail2 = count_suc3 = count_fail3 = count_suc4 = count_fail4 = 0

# ================= Coefficient testing ================= #
for i in range(0, N):  # for 100 trials
    vessel = Ship(D = 40)

    bot1 = Bot1(copy.deepcopy(vessel))
    if bot1.mission_success(0.5)[0]:
        count_suc1 += 1
    else:
        count_fail1 += 1
        
    bot2 = Bot2(copy.deepcopy(vessel))
    if bot2.mission_success(0.5)[0]:
        count_suc2 += 1
    else:
        count_fail2 += 1
        
    bot3 = Bot3(copy.deepcopy(vessel))
    if bot3.mission_success(0.5)[0]:
        count_suc3 += 1
    else:
        count_fail3 += 1

print(count_suc1)
print(count_suc2)
print(count_suc3)

for factor in range(1, 10):
    count_suc4 = 0
    for i in range(0, N):  # for 100 trials
        vessel = Ship(D = 40)
        bot4 = Bot4(copy.deepcopy(vessel))
        if bot4.mission_success(0.5, factor)[0]:
            count_suc4 += 1
        else:
            count_fail4 += 1
    print(factor, count_suc4)

# Question 3
# ================= Perform trials ================= #
for q in q_values:  # for each q
    count_suc1 = 0
    count_suc2 = 0
    count_suc3 = 0
    count_suc4 = 0

    for i in range(0, N):  # for 100 trials
        vessel = Ship(D = 40)

        bot1 = Bot1(copy.deepcopy(vessel))
        if bot1.mission_success(q)[0]:
            count_suc1 += 1
        else:
            count_fail1 += 1
            
        bot2 = Bot2(copy.deepcopy(vessel))
        if bot2.mission_success(q)[0]:
            count_suc2 += 1
        else:
            count_fail2 += 1
            
        bot3 = Bot3(copy.deepcopy(vessel))
        if bot3.mission_success(q)[0]:
            count_suc3 += 1
        else:
            count_fail3 += 1
        
        bot4 = Bot4(copy.deepcopy(vessel))
        if bot4.mission_success(q, 1)[0]:
            count_suc4 += 1
        else:
            count_fail4 += 1

    bot1_q_successes.append(count_suc1/N)
    bot2_q_successes.append(count_suc2/N)
    bot3_q_successes.append(count_suc3/N)
    bot4_q_successes.append(count_suc4/N)

# ================= Create plots ================= #
plt.figure(figsize = (12, 8))

# Plot success rates for each bot
plt.plot(q_values, bot1_q_successes, marker='o', label = "Bot1")
plt.plot(q_values, bot2_q_successes, marker='o', label = "Bot2")
plt.plot(q_values, bot3_q_successes, marker='o', label = "Bot3")
plt.plot(q_values, bot4_q_successes, marker='o', label = "Bot4")
plt.grid(True, linestyle='--')

plt.xlabel('Fire Spread Probability (q)')
plt.ylabel('Success Rate')
plt.title('Bot Success Rate vs. Fire Spread Probability (q)')
plt.legend()

plt.show()

# # Question 4
# # Run simulations for different values of q
# winnable_frequencies = []
# num_trials = 25

# for q in q_values:
#     winnable_count = 0
#     for trial in range(num_trials):
#         vessel = Ship(D = 40)
        
#         #create the same seed for each bot in order to make the sims deterministic
#         random.seed(trial)
#         bot1 = Bot1(copy.deepcopy(vessel))
#         random.seed(trial)
#         bot2 = Bot2(copy.deepcopy(vessel))
#         random.seed(trial)
#         bot3 = Bot3(copy.deepcopy(vessel))
#         random.seed(trial)
#         bot4 = Bot4(copy.deepcopy(vessel))
        
#         #if any of the bots succeed then the simulation is winnable
#         success = (bot1.mission_success(q)[0] or 
#                       bot2.mission_success(q)[0] or 
#                       bot3.mission_success(q)[0] or 
#                       bot4.mission_success(q, 1)[0])
        
#         if success:
#             winnable_count += 1
#             continue
        
#         print("Q:" + str(q))
#         # print("Bot1: " + str(bot1.get_timestep_count()) + " | " + str(bot1.get_visited_positions()))
#         # print(bot1.SHIP)
#         # print()
#         # print("Bot2: " + str(bot2.get_timestep_count()) + " | " + str(bot2.get_visited_positions()))
#         # print(bot2.SHIP)
#         # print()
#         # print("Bot3: " + str(bot3.get_timestep_count()) + " | " + str(bot3.get_visited_positions()))
#         # print(bot3.SHIP)
#         # print()
#         # print("Bot4: " + str(bot4.get_timestep_count()) + " | " + str(bot4.get_visited_positions()))
#         # print(bot4.SHIP)
#         # print()
#         # print("\n\n")
        
#         #find the bot that went the furthest before failing 
#         max_timestep = len(bot1.get_visited_positions())
#         max_bot = bot1
#         for bot in [bot2, bot3, bot4]:
#             if max_timestep < len(bot.get_visited_positions()):
#                 max_timestep = len(bot.get_visited_positions())
#                 max_bot = bot
#         # print("" + str(max_bot) + " has max length of " + str(max_timestep))
        
#         random.seed(trial)
#         sim = Winnability(vessel, max_bot.SHIP, max_timestep, q)
#         viable_solution = sim.is_winnable()
#         if viable_solution: #if the list returned is not empty then it found a viable path 
#             # print("~~~~~~~~~~~~~~~~THERE IS STILL A WINNABLE SOLUTION~~~~~~~~~~~~~~~~")
#             # print(viable_solution)
#             # print(sim.final_ship)
            
           
#             # # Define paths for each bot and the "Perfect" path
#             # paths = {
#             #     "Bot1": bot1.get_visited_positions(),
#             #     "Bot2": bot2.get_visited_positions(),
#             #     "Bot3": bot3.get_visited_positions(),
#             #     "Bot4": bot4.get_visited_positions(),
#             #     "Perfect": viable_solution
#             # }

#             # # Define a colormap for the grid (Closed = Black, Open = White, Fire = Orange)
#             # cmap = mcolors.ListedColormap(["black", "white", "orange"])
#             # bounds = [0, 1, 3, 4]  # Boundaries for values
#             # norm = mcolors.BoundaryNorm(bounds, cmap.N)

#             # # Define path colors
#             # path_colors = {"Bot1": "red", "Bot2": "blue", "Bot3": "purple", "Bot4": "green", "Perfect": "turquoise"}

#             # # Create a figure with subplots (2 rows, 3 columns for 5 plots)
#             # fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#             # axes = axes.flatten()  # Flatten for easy iteration

#             # # Plot each bot's path separately
#             # for i, (bot, path) in enumerate(paths.items()):
#             #     ax = axes[i]
                
#             #     # Plot the base grid
#             #     ax.imshow(sim.final_ship.grid, cmap=cmap, norm=norm, origin="upper")
                
#             #     # Extract x, y coordinates and plot the path
#             #     x, y = zip(*path)
#             #     ax.plot(y, x, color=path_colors[bot], linewidth=5, label=f"{bot} Path")
                
#             #     # Formatting
#             #     ax.set_title(f"{bot} Path")
#             #     ax.legend()

#             # # remove last subplot as we only use 5
#             # fig.delaxes(axes[5])

#             # # Adjust layout and show
#             # plt.suptitle("Path Visualization", fontsize=16)
#             # plt.tight_layout()
#             # plt.show()
#             winnable_count += 1
            
#     winnable_frequencies.append(winnable_count / num_trials)

# # Plot results
# # Plot simple winnability results
# plt.figure(figsize=(10, 6))

# # Plot winnability curve with markers
# plt.plot(q_values, winnable_frequencies, marker='o', linestyle='-', color='blue')

# # Add grid
# plt.grid(True)

# # Add labels and title
# plt.xlabel('Flammability (q)')
# plt.ylabel('Probability of Winnability')
# plt.title('Winnability of Simulation as a Function of Fire Spread Probability')

# # Set y-axis limits
# plt.ylim(0, 1)

# # Show plot
# plt.show()







# # # Used to test best coefficients for heuristic
# # for factor in range(1,12):
# #     count_suc4 = 0
# #     count_fail4 = 0
    
# #     for i in range(0, 100):

# #         vessel = Ship(D = 15)
# #         bot4 = Bot4(copy.deepcopy(vessel))

# #         if bot4.mission_success(1, factor):
# #             count_suc4 += 1
# #         else:
# #             count_fail4 += 1

# #     print(f"Bot4: {factor}") 
# #     print("Successes: ", count_suc4)
# #     print("Failures: ", count_fail4)