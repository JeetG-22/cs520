import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from ship import Ship
from bot import Bot
import time

class SimpleModel(nn.Module):

    def __init__(self, input_size):
        super(SimpleModel, self).__init__()
        self.layer1 = nn.Linear(input_size, 64)
        self.layer2 = nn.Linear(64,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return x

def create_feature_vect(ship, L):
    grid = np.zeros(ship.N * ship.N)
    for position in  L.keys():
        index = position[0] * ship.N + position[1]
        if index < len(grid):
            grid[index] = 1
    return np.append(len(L) / len(ship.open_cells), grid)

# Generate random initial belief state
def make_random_L(ship, size):
    cells = random.sample(list(ship.open_cells.keys()), size)
    return {cell: cell for cell in cells}

def simulate_move(ship, L, direction):
    # What happens if move succeeds
    L_success = {}
    # What happens if move fails
    L_fail = {}
    
    for pos, curr_pos in L.items():
        new_x = curr_pos[0] + direction[0]
        new_y = curr_pos[1] + direction[1]
        new_pos = (new_x, new_y)
        
        # Check if move would succeed
        if new_pos in ship.open_cells:
            L_success[pos] = new_pos
        else:
            L_fail[pos] = curr_pos
    
    return L_success, L_fail

# Strategy π0: use implementation from part 2
def strategy_pi0(ship, L):
    try:
        bot_instance = Bot(ship)
        
        #get_moves from part2
        final_pos, moves_count = bot_instance.get_moves(L.copy())
        
        return moves_count
    
    except Exception as e:
        return 50  # Return a moderate value if there's an error

def collect_pi0_data(num_samples, ship_size):
    
    X = []  # features
    y = []  # moves needed
    
    for i in range(num_samples):
        ship = Ship(ship_size)
        ship.place_entities()
        
        L_size = random.randint(2, len(ship.open_cells))
        L = make_random_L(ship, L_size)
        
        try:
            moves = strategy_pi0(ship, L)
            
            X.append(create_feature_vect(ship, L))
            y.append(moves)

        except Exception as e:
            print(f"Error {e}")
            continue
    
    return np.array(X), np.array(y)

# model to predict moves needed
def train_model(X, y, epochs=20):
    
    # split into train/test
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # convert PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    model = SimpleModel(X.shape[1])
    
    # loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    train_losses = []
    test_losses = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_losses.append(test_loss)
        
        print(f"Epoch {epoch}\n Training Loss: {loss.item()}\n Test Loss: {test_loss}")
    
    # plot training progress
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return model

# use model to predict moves
def predict_moves(model, ship, L):
    features = create_feature_vect(ship, L)
    features_tensor = torch.tensor([features], dtype=torch.float32)
    
    with torch.no_grad():
        prediction = model(features_tensor).item()
    
    return max(1, prediction)  # make sure we predict at least 1 move

# strategy π1: model used to predict best direction
def strategy_pi1(model, ship, L):
    try:
        # For really small L, just use original strategy
        if len(L) <= 2:
            return strategy_pi0(ship, L)
            
        directions = [(0,-1), (1,0), (0,1), (-1,0)]  
        direction_names = ["Up", "Right", "Down", "Left"]
        expected_moves = []
                
        for i, direction in enumerate(directions):
            # simulate what happens after this move
            L_success, L_fail = simulate_move(ship, L, direction)
            
            # calculate probability of success
            p_success = len(L_success) / len(L) if L_success else 0
            p_fail = 1 - p_success
            
            # skip if this direction doesn't reduce uncertainty
            if L_success and L_fail and len(L_success) == len(L) and len(L_fail) == len(L):
                expected_moves.append(float('inf'))
                continue
                
            # use model to predict moves for each outcome
            moves_success = predict_moves(model, ship, L_success) if L_success else 0
            moves_fail = predict_moves(model, ship, L_fail) if L_fail else 0
            
            # expected moves = 1 (for this move) + weighted average of outcomes
            total_expected = 1 + p_success * moves_success + p_fail * moves_fail
            expected_moves.append(total_expected)
        
        # to find best directtion
        if expected_moves and min(expected_moves) < float('inf'):
            best_idx = expected_moves.index(min(expected_moves))
            
            return expected_moves[best_idx]
        
        # use original strategy if model cant find best directon
        return strategy_pi0(ship, L)
    
    except Exception as e:
        print(f"Error in π1: {e}")
        # If there's an error, try original strategy
        try:
            return strategy_pi0(ship, L)
        except:
            return 50  # Default value if both strategies fail

# Compare strategies
def compare_strategies(model, ship_size=10, num_tests=5):
    L_sizes = []
    pi0_moves = []
    pi1_moves = []
    
    # different L sizes
    L_size_options = [3, 5, 8, 12, 15]  
    for L_size in L_size_options:        
        pi0_total = 0
        pi1_total = 0
        success_count = 0
        
        for i in range(num_tests): #multitesting
            try:
                ship = Ship(ship_size)
                ship.place_entities()
                L = make_random_L(ship, L_size)
                
                # copy of L for each strategy
                L0 = {k: v for k, v in L.items()}
                L1 = {k: v for k, v in L.items()}
                
                # testing strategies using identical starting conditions
                moves_pi0 = strategy_pi0(ship, L0)
                moves_pi1 = strategy_pi1(model, ship, L1)
                
                pi0_total += moves_pi0
                pi1_total += moves_pi1
                success_count += 1
            
            except Exception as e:
                print(f"Error {e}")
        
        # record only when successful
        if success_count > 0:
            # Calculate averages
            pi0_avg = pi0_total / success_count
            pi1_avg = pi1_total / success_count
            
            L_sizes.append(L_size)
            pi0_moves.append(pi0_avg)
            pi1_moves.append(pi1_avg)
            
            # to calculate improvement
            if pi1_avg < pi0_avg:
                improvement = (pi0_avg - pi1_avg) / pi0_avg * 100
                print(f"Improvement: {improvement} for |L|: {L_size}")
            else:
                print(f"No improvement for {L_size}")
    
    # plotting
    if L_sizes:
        plt.plot(L_sizes, pi0_moves, label='π0')
        plt.plot(L_sizes, pi1_moves, label='π1')
        plt.xlabel('Size of L')
        plt.ylabel('Average Moves Needed')
        plt.title('Strategy Comparison')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # improvements
        improvements = [(pi0 - pi1) / pi0 * 100 for pi0, pi1 in zip(pi0_moves, pi1_moves) if pi0 > 0]
        if improvements:
            overall_improvement = sum(improvements) / len(improvements)
            print(f"\nImprovement: {overall_improvement}%")
        else:
            print("\nNo improvement")
            
        # generate data for plot with |L| on x-axis
        plot_L_vs_moves(model, ship_size)

def plot_L_vs_moves(model, ship_size, num_samples=3):
    ship = Ship(ship_size)
    ship.place_entities()
    
    L_sizes = range(2, len(ship.open_cells), max(1, len(ship.open_cells) // 10))
    pi0_avg_moves = []
    pi1_avg_moves = []
    
    for L_size in L_sizes:
        pi0_moves = []
        pi1_moves = []
        
        for _ in range(num_samples):
            try:
                L = make_random_L(ship, L_size)
                
                moves_pi0 = strategy_pi0(ship, L.copy())
                moves_pi1 = strategy_pi1(model, ship, L.copy())
                
                pi0_moves.append(moves_pi0)
                pi1_moves.append(moves_pi1)
            except Exception:
                continue
        
        # averages calculation
        if pi0_moves and pi1_moves:
            pi0_avg_moves.append(sum(pi0_moves) / len(pi0_moves))
            pi1_avg_moves.append(sum(pi1_moves) / len(pi1_moves))
        else:
            # skip if we don't have data
            continue
    
    # create plot
    plt.plot(L_sizes[:len(pi0_avg_moves)], pi0_avg_moves, label='π0')
    plt.plot(L_sizes[:len(pi1_avg_moves)], pi1_avg_moves, label='π1')
    plt.xlabel('Size of L')
    plt.ylabel('Average number of moves')
    plt.title('|L| vs. Num Moves')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    ship_size = 10
    num_samples = 50
    num_tests = 10
    
    # collect data from original strategy
    X, y = collect_pi0_data(num_samples, ship_size)
    
    # only continue if we have data
    if len(X) > 0:
        model = train_model(X, y)
        compare_strategies(model, ship_size, num_tests)
    else:
        print("Could not collect enough data")

if __name__ == "__main__":
    main()