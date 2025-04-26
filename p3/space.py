import ship

# Note: ship does not have closed cells bordering its edges for some reason
ship = ship.Ship(6)
ship.place_entities()
print(ship)
