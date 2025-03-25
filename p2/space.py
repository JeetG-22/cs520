import ship
import baseline_bot
N = 5

# Dimensions set to 30x30
spaceship = ship.Ship(N)

# Block all outer edge cells and pop them from open cells list
for i in range(0, N):
    spaceship.grid[0][i] = 0
    if (0, i) in spaceship.open_cells:
        spaceship.open_cells.pop((0, i))
    spaceship.grid[i][0] = 0
    if (i, 0) in spaceship.open_cells:
        spaceship.open_cells.pop((i, 0))
    spaceship.grid[N-1][i] = 0
    if (N-1, i) in spaceship.open_cells:
        spaceship.open_cells.pop((N-1, i))
    spaceship.grid[i][N-1] = 0
    if (i, N-1) in spaceship.open_cells:
        spaceship.open_cells.pop((i, N-1))

# Place bot and rat
spaceship.place_entities()

print(spaceship.grid)

bot = baseline_bot.Baseline(spaceship)
print(bot.get_position(2))
print(bot.get_est_pos(bot.get_position(2)))