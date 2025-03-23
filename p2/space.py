import ship

# Dimensions set to 30x30
spaceship = ship.Ship()

# Block all outer edge cells and pop them from open cells list
for i in range(0, 30):
    spaceship.grid[0][i] = 0
    if (0, i) in spaceship.open_cells:
        spaceship.open_cells.pop((0, i))
    spaceship.grid[i][0] = 0
    if (i, 0) in spaceship.open_cells:
        spaceship.open_cells.pop((i, 0))
    spaceship.grid[29][i] = 0
    if (29, i) in spaceship.open_cells:
        spaceship.open_cells.pop((29, i))
    spaceship.grid[i][29] = 0
    if (i, 29) in spaceship.open_cells:
        spaceship.open_cells.pop((i, 29))

# Place bot
spaceship.place_entities()

print(spaceship.grid)