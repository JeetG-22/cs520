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
        if bot4.mission_success(q, 1):
            count_suc4 += 1
        else:
            count_fail4 += 1

    bot1_q_successes.append(count_suc1/N)
    bot2_q_successes.append(count_suc2/N)
    bot3_q_successes.append(count_suc3/N)
    bot4_q_successes.append(count_suc4/N)

# Create plots
plt.figure(figsize = (12, 8))

# Plot success rates for each bot with different colors and markers
plt.plot(q_values, bot1_q_successes, marker='o', label = "Bot1")
plt.plot(q_values, bot2_q_successes, marker='o', label = "Bot2")
plt.plot(q_values, bot3_q_successes, marker='o', label = "Bot3")
plt.plot(q_values, bot4_q_successes, marker='o', label = "Bot4")

# Set y-axis limit

plt.grid(True, linestyle='--')

# Labels and title
plt.xlabel('Fire Spread Probability(q)', fontsize=14)
plt.ylabel('Success Rate', fontsize=14)
plt.title('Bot Success Rate vs. Fire Spread Probability(q)', fontsize=16)

# Add legend
plt.legend()

# Show plot
plt.show()


# Question 4 & 5
# ================= Winnability Probability ================= #
# Run simulations for different values of q
winnable_frequencies = []
num_trials = 100

# # Initialize tracking variables
bot1_winnable_success_deter = []
bot2_winnable_success_deter = []
bot3_winnable_success_deter = []
bot4_winnable_success_deter = []

for q in q_values:
    winnable_count = 0
    bot1_wins = 0
    bot2_wins = 0
    bot3_wins = 0
    bot4_wins = 0
    
    for trial in range(num_trials):
        vessel = Ship(D = 40)
        seed_value = hash((q, trial))
        success = False
        
        #create the same seed for each bot in order to make the sims deterministic
        random.seed(seed_value)
        bot1 = Bot1(copy.deepcopy(vessel))
        if bot1.mission_success(q)[0]:
            bot1_wins += 1
            success = True
            
        random.seed(seed_value)
        bot2 = Bot2(copy.deepcopy(vessel))
        if bot2.mission_success(q)[0]:
            bot2_wins += 1
            success = True
            
        random.seed(seed_value)
        bot3 = Bot3(copy.deepcopy(vessel))
        if bot3.mission_success(q)[0]:
            bot3_wins += 1
            success = True
            
        random.seed(seed_value)
        bot4 = Bot4(copy.deepcopy(vessel))
        if bot4.mission_success(q, 1)[0]:
            bot4_wins += 1
            success = True
        
        if success:
            winnable_count += 1
            continue
        
        print("Q:" + str(q))
        #find the bot that went the furthest before failing 
        max_timestep = len(bot1.get_visited_positions())
        max_bot = bot1
        for bot in [bot2, bot3, bot4]:
            if max_timestep < len(bot.get_visited_positions()):
                max_timestep = len(bot.get_visited_positions())
                max_bot = bot
        
        random.seed(seed_value)
        sim = Winnability(vessel, max_bot.SHIP, max_timestep, q)
        viable_solution = sim.is_winnable()
        if viable_solution and len(viable_solution) <= max_timestep: #if the list returned is not empty then it found a viable path 
            winnable_count += 1
            
            # Create a plot mapping out the paths of the different bots and perfect path 
            paths = {
                "Bot1": bot1.get_visited_positions(),
                "Bot2": bot2.get_visited_positions(),
                "Bot3": bot3.get_visited_positions(),
                "Bot4": bot4.get_visited_positions(),
                "Perfect": viable_solution
            }

            # Define a colormap for the grid (Closed = Black, Open = White, Fire = Orange)
            cmap = mcolors.ListedColormap(["black", "white", "orange"])
            bounds = [0, 1, 3, 4]  # Boundaries for values
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Define path colors
            path_colors = {"Bot1": "red", "Bot2": "blue", "Bot3": "purple", "Bot4": "green", "Perfect": "turquoise"}

            # Create a figure with subplots (2 rows, 3 columns for 5 plots)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()  # Flatten for easy iteration

            # Plot each bot's path separately
            for i, (bot, path) in enumerate(paths.items()):
                ax = axes[i]
                
                # Plot the base grid
                ax.imshow(sim.final_ship.grid, cmap=cmap, norm=norm, origin="upper")
                
                # Extract x, y coordinates and plot the path
                x, y = zip(*path)
                ax.plot(y, x, color=path_colors[bot], linewidth=5, label=f"{bot} Path")
                
                # Formatting
                ax.set_title(f"{bot} Path")
                ax.legend()

            # remove last subplot as we only use 5
            fig.delaxes(axes[5])

            # Adjust layout and show
            plt.suptitle("Path Visualization", fontsize=16)
            plt.tight_layout()
            plt.show()
            
    winnable_frequencies.append(winnable_count / num_trials)
    if winnable_count != 0:
        bot1_winnable_success_deter.append(bot1_wins / winnable_count)
        bot2_winnable_success_deter.append(bot2_wins / winnable_count)
        bot3_winnable_success_deter.append(bot3_wins / winnable_count)
        bot4_winnable_success_deter.append(bot4_wins / winnable_count)
    else:
        bot1_winnable_success_deter.append(0)
        bot2_winnable_success_deter.append(0)
        bot3_winnable_success_deter.append(0)
        bot4_winnable_success_deter.append(0)

# Plot results
plt.figure(figsize=(10, 6))

plt.plot(q_values, winnable_frequencies, marker='o', linestyle='-', color='blue')

# Add grid
plt.grid(True)

# Add labels and title
plt.xlabel('Flammability (q)')
plt.ylabel('Probability of Winnability')
plt.title('Winnability of Simulation as a Function of Fire Spread Probability')

# Show plot
plt.show()

# Initialize tracking variables
bot1_winnable_success_random = []
bot2_winnable_success_random = []
bot3_winnable_success_random = []
bot4_winnable_success_random = []

for q in q_values:
    winnable_count = 0
    bot1_wins = 0
    bot2_wins = 0
    bot3_wins = 0
    bot4_wins = 0
    
    for trial in range(num_trials):
        vessel = Ship(D = 40)
        success = False

        bot1 = Bot1(copy.deepcopy(vessel))
        if bot1.mission_success(q)[0]:
            bot1_wins += 1
            success = True
            
        bot2 = Bot2(copy.deepcopy(vessel))
        if bot2.mission_success(q)[0]:
            bot2_wins += 1
            success = True
            
        bot3 = Bot3(copy.deepcopy(vessel))
        if bot3.mission_success(q)[0]:
            bot3_wins += 1
            success = True
            
        bot4 = Bot4(copy.deepcopy(vessel))
        if bot4.mission_success(q, 1)[0]:
            bot4_wins += 1
            success = True
        
        if success:
            winnable_count += 1
            continue
        
        print("Q:" + str(q))
        #find the bot that went the furthest before failing 
        max_timestep = len(bot1.get_visited_positions())
        max_bot = bot1
        for bot in [bot2, bot3, bot4]:
            if max_timestep < len(bot.get_visited_positions()):
                max_timestep = len(bot.get_visited_positions())
                max_bot = bot
        
        sim = Winnability(vessel, max_bot.SHIP, max_timestep, q)
        viable_solution = sim.is_winnable()
        if viable_solution and len(viable_solution) <= max_timestep: #if the list returned is not empty then it found a viable path 
            winnable_count += 1
            
            # Define paths for each bot and the "Perfect" path
            paths = {
                "Bot1": bot1.get_visited_positions(),
                "Bot2": bot2.get_visited_positions(),
                "Bot3": bot3.get_visited_positions(),
                "Bot4": bot4.get_visited_positions(),
                "Perfect": viable_solution
            }

            # Define a colormap for the grid (Closed = Black, Open = White, Fire = Orange)
            cmap = mcolors.ListedColormap(["black", "white", "orange"])
            bounds = [0, 1, 3, 4]  # Boundaries for values
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Define path colors
            path_colors = {"Bot1": "red", "Bot2": "blue", "Bot3": "purple", "Bot4": "green", "Perfect": "turquoise"}

            # Create a figure with subplots (2 rows, 3 columns for 5 plots)
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()  # Flatten for easy iteration

            # Plot each bot's path separately
            for i, (bot, path) in enumerate(paths.items()):
                ax = axes[i]
                
                # Plot the base grid
                ax.imshow(sim.final_ship.grid, cmap=cmap, norm=norm, origin="upper")
                
                # Extract x, y coordinates and plot the path
                x, y = zip(*path)
                ax.plot(y, x, color=path_colors[bot], linewidth=5, label=f"{bot} Path")
                
                # Formatting
                ax.set_title(f"{bot} Path")
                ax.legend()

            # remove last subplot as we only use 5
            fig.delaxes(axes[5])

            # Adjust layout and show
            plt.suptitle("Path Visualization", fontsize=16)
            plt.tight_layout()
            plt.show()
            
    if winnable_count != 0:
        bot1_winnable_success_random.append(bot1_wins / winnable_count)
        bot2_winnable_success_random.append(bot2_wins / winnable_count)
        bot3_winnable_success_random.append(bot3_wins / winnable_count)
        bot4_winnable_success_random.append(bot4_wins / winnable_count)
    else:
        bot1_winnable_success_random.append(0)
        bot2_winnable_success_random.append(0)
        bot3_winnable_success_random.append(0)
        bot4_winnable_success_random.append(0)

# Create plot
fig, axs = plt.subplots(1,2, figsize=(15, 10))
axs = axs.flatten()

# Plot success rates for each bot deterministic
axs[0].plot(q_values, bot1_winnable_success_deter, marker='o', label="Bot1", color='blue')
axs[0].plot(q_values, bot2_winnable_success_deter, marker='o', label="Bot2", color='orange')
axs[0].plot(q_values, bot3_winnable_success_deter, marker='o', label="Bot3", color='green')
axs[0].plot(q_values, bot4_winnable_success_deter, marker='o', label="Bot4", color='red')

# Add grid
axs[0].grid(True)

# Labels and title
axs[0].set_xlabel('Flammability (q)')
axs[0].set_ylabel('Success Rate on Winnable Simulations')
axs[0].set_title('Bot Performance on Winnable Simulations With Deterministic Fire Spread')

# Add legend
axs[0].legend()

# Plot success rates for each bot random
axs[1].plot(q_values, bot1_winnable_success_random, marker='o', label="Bot1", color='blue')
axs[1].plot(q_values, bot2_winnable_success_random, marker='o', label="Bot2", color='orange')
axs[1].plot(q_values, bot3_winnable_success_random, marker='o', label="Bot3", color='green')
axs[1].plot(q_values, bot4_winnable_success_random, marker='o', label="Bot4", color='red')

# Add grid
axs[1].grid(True)

# Labels and title
axs[1].set_xlabel('Flammability (q)')
axs[1].set_ylabel('Success Rate on Winnable Simulations')
axs[1].set_title('Bot Performance on Winnable Simulations With Random Fire Spread')

# Add legend
axs[1].legend()

# Show plot
plt.show()


# Question 6
# ================= Why Bots Fail? ================= #
for q in q_values:
    
    for trial in range(num_trials):
        vessel = Ship(D = 40)
        seed_value = hash((q, trial))
        success = False
        bot_fails = False
        successful_bots = []
        
        #create the same seed for each bot in order to make the sims deterministic
        random.seed(seed_value)
        bot1 = Bot1(copy.deepcopy(vessel))
        bot1_success = bot1.mission_success(q)[0]
        if bot1_success:
            success = True
            successful_bots.append(bot1)
        else:
            bot_fails = True
            
        random.seed(seed_value)
        bot2 = Bot2(copy.deepcopy(vessel))
        bot2_success = bot2.mission_success(q)[0]
        if bot2_success:
            success = True
            successful_bots.append(bot2)
        else:
            bot_fails = True
            
        random.seed(seed_value)
        bot3 = Bot3(copy.deepcopy(vessel))
        bot3_success = bot3.mission_success(q)[0]
        if bot3_success:
            success = True
            successful_bots.append(bot3)
        else:
            bot_fails = True
            
        random.seed(seed_value)
        bot4 = Bot4(copy.deepcopy(vessel))
        bot4_success = bot4.mission_success(q)[0]
        if bot4_success:
            success = True
            successful_bots.append(bot4)
        else:
            bot_fails = True
        
        if success and bot_fails:
            #find the bot that went the furthest before failing 
            min_timestep = len(successful_bots[0].get_visited_positions())
            min_bot = successful_bots[0]
            for bot in successful_bots:
                if min_timestep > len(bot.get_visited_positions()):
                    min_timestep = len(bot.get_visited_positions())
                    min_bot = bot
            print(successful_bots)
                
            # Define paths for each bot and the "Perfect" path
            paths = {
                "Bot1": bot1.get_visited_positions(),
                "Bot2": bot2.get_visited_positions(),
                "Bot3": bot3.get_visited_positions(),
                "Bot4": bot4.get_visited_positions(),
            }

            # Define a colormap for the grid (Closed = Black, Open = White, Fire = Orange)
            cmap = mcolors.ListedColormap(["black", "white", "orange"])
            bounds = [0, 1, 3, 4]  # Boundaries for values
            norm = mcolors.BoundaryNorm(bounds, cmap.N)

            # Define path colors
            path_colors = {"Bot1": "red", "Bot2": "blue", "Bot3": "purple", "Bot4": "green"}

            # Create a figure with subplots (2 rows, 3 columns for 5 plots)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.flatten()  # Flatten for easy iteration

            # Plot each bot's path separately
            for i, (bot, path) in enumerate(paths.items()):
                ax = axes[i]
                
                # Plot the base grid
                ax.imshow(min_bot.SHIP.grid, cmap=cmap, norm=norm, origin="upper")
                
                # Extract x, y coordinates and plot the path
                x, y = zip(*path)
                ax.plot(y, x, color=path_colors[bot], linewidth=5, label=f"{bot} Path")
                
                # Formatting
                ax.set_title(f"{bot} Path")
                ax.legend()

            # Adjust layout and show
            plt.suptitle("Path Visualization", fontsize=16)
            plt.tight_layout()
            plt.show()
            