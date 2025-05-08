import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from ship import Ship
from bot import Bot
import time

random.seed(42)
torch.manual_seed(42)

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

# Simulate what happens after a move
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

def collect_pi0_data(num_samples=100, ship_size=10):
    print(f"Collecting {num_samples} data points from Part 2 strategy...")
    
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
            
            if (i+1) % 10 == 0:
                print(f"  Collected {i+1} samples")
        except Exception as e:
            print(f"  Error with sample {i+1}: {e}")
            continue
    
    return np.array(X), np.array(y)

# model to predict moves needed
def train_model(X, y, epochs=20):
    print("Training model...")
    
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
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}")
    
    # plot training progress
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Progress')
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
        
        print(f"Evaluating directions for state with |L| = {len(L)}:")
        
        for i, direction in enumerate(directions):
            # simulate what happens after this move
            L_success, L_fail = simulate_move(ship, L, direction)
            
            # calculate probability of success
            p_success = len(L_success) / len(L) if L_success else 0
            p_fail = 1 - p_success
            
            # skip if this direction doesn't reduce uncertainty
            if L_success and L_fail and len(L_success) == len(L) and len(L_fail) == len(L):
                expected_moves.append(float('inf'))
                print(f"  {direction_names[i]}: No uncertainty reduction, skipping")
                continue
                
            # use model to predict moves for each outcome
            moves_success = predict_moves(model, ship, L_success) if L_success else 0
            moves_fail = predict_moves(model, ship, L_fail) if L_fail else 0
            
            # expected moves = 1 (for this move) + weighted average of outcomes
            total_expected = 1 + p_success * moves_success + p_fail * moves_fail
            expected_moves.append(total_expected)
            
            print(f"  {direction_names[i]}: Expected {total_expected:.2f} moves " +
                  f"(Success: {p_success:.2f}→{len(L_success) if L_success else 0} cells, " +
                  f"Fail: {p_fail:.2f}→{len(L_fail) if L_fail else 0} cells)")
        
        # to find best directtion
        if expected_moves and min(expected_moves) < float('inf'):
            best_idx = expected_moves.index(min(expected_moves))
            print(f"Best direction: {direction_names[best_idx]} with {expected_moves[best_idx]:.2f} expected moves")
            
            return expected_moves[best_idx]
        
        # use original strategy if model cant find best directon
        print("No good direction found, using original strategy")
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
    print(f"Comparing strategies on {num_tests} tests per L size...")
    
    L_sizes = []
    pi0_moves = []
    pi1_moves = []
    
    # different L sizes
    L_size_options = [3, 5, 8, 12, 15]  
    for L_size in L_size_options:
        print(f"\nTesting |L| = {L_size}:")
        
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
                
                print(f"  Test {i+1}: π0={moves_pi0}, π1={moves_pi1}")
            except Exception as e:
                print(f"  Error in test {i+1}: {e}")
        
        # record only when successful
        if success_count > 0:
            # Calculate averages
            pi0_avg = pi0_total / success_count
            pi1_avg = pi1_total / success_count
            
            L_sizes.append(L_size)
            pi0_moves.append(pi0_avg)
            pi1_moves.append(pi1_avg)
            
            print(f"Average for |L|={L_size}: π0={pi0_avg:.2f}, π1={pi1_avg:.2f}")
            
            # to calculate improvement
            if pi1_avg < pi0_avg:
                improvement = (pi0_avg - pi1_avg) / pi0_avg * 100
                print(f"π1 is {improvement:.2f}% better!")
            else:
                print("π1 didn't improve over π0 for this size.")
    
    # plotting
    if L_sizes:
        plt.figure(figsize=(10, 6))
        plt.plot(L_sizes, pi0_moves, 'bo-', label='Original Strategy (π0)')
        plt.plot(L_sizes, pi1_moves, 'ro-', label='Improved Strategy (π1)')
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
            print(f"\nOverall, π1 improved over π0 by {overall_improvement:.2f}%")
        else:
            print("\nπ1 did not show overall improvement over π0.")
            
        # generate data for plot with |L| on x-axis
        print("\nGenerating plot data for different |L| values...")
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
    plt.figure(figsize=(10, 6))
    plt.plot(L_sizes[:len(pi0_avg_moves)], pi0_avg_moves, 'bo-', label='Original Strategy (π0)')
    plt.plot(L_sizes[:len(pi1_avg_moves)], pi1_avg_moves, 'ro-', label='Improved Strategy (π1)')
    plt.xlabel('Size of L (number of possible locations)')
    plt.ylabel('Average number of moves needed')
    plt.title('Localization Performance: |L| vs. Moves')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    ship_size = 10
    num_samples = 50
    num_tests = 10  # Keep small for demonstration
    
    # collect data from original strategy
    X, y = collect_pi0_data(num_samples, ship_size)
    
    # only continue if we have data
    if len(X) > 0:
        model = train_model(X, y)
        
        compare_strategies(model, ship_size, num_tests)
    else:
        print("Error: Could not collect enough data. Please check your Bot implementation.")

if __name__ == "__main__":
    main()