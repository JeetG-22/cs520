import ship

# Dimensions set to 30x30
spaceship = ship.Ship()

# Block all outer edge cells
for i in range(0, 30):
    spaceship.grid[0][i] = 0
    spaceship.grid[i][0] = 0
    spaceship.grid[29][i] = 0
    spaceship.grid[i][29] = 0

# Place entities
spaceship.place_entities()
