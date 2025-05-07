import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import random
from ship import Ship
from bot import Bot
import time

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

class SimpleLocalizationModel(nn.Module):
    """
    Simple neural network model for predicting moves needed for localization.
    Similar to the MNIST model in the tutorial, but for regression.
    """
    def __init__(self, input_size):
        super(SimpleLocalizationModel, self).__init__()
        
        # Define a simple architecture with one hidden layer
        self.layers = nn.Sequential(
            nn.Linear(input_size, 64),  # Input -> Hidden
            nn.ReLU(),                  # Activation function
            nn.Linear(64, 1)            # Hidden -> Output
        )
    
    def forward(self, x):
        # Forward pass through the network
        return self.layers(x)


def collect_data(num_samples=200, ship_size=10):
    """
    Collect data by measuring the relationship between L size and moves needed.
    This function generates training data for our ML model.
    """
    print(f"Generating {num_samples} data points...")
    
    features_list = []  # input features
    moves_list = []  # target values (moves needed)
    
    for _ in range(num_samples):
        # new ship config for each sample
        ship = Ship(ship_size)
        ship.place_entities()
        
        # pick random # of cells for belief state
        max_cells = len(ship.open_cells)
        num_cells = random.randint(2, max_cells)
        
        # create a random belief state
        L = {}
        open_cells = list(ship.open_cells.keys())
        selected_cells = random.sample(open_cells, num_cells) #to get a sample of a random # of open cells
        for cell in selected_cells:
            L[cell] = cell
        
        
        # get estimate moves from random belief state
        moves = simulate_localization(ship, L)
        
        # Create a feature vector
        grid = np.zeros(ship_size * ship_size) #create L to a grid like MNIST images
        for pos in L.keys():
            x,y = pos
            index = x * ship_size + y
            if index < len(grid): 
                grid[index] = 1
        
        # Combine features
        features = np.append(num_cells / max_cells, grid)
        
        # Add to our dataset
        features_list.append(features)
        moves_list.append(moves)
    
    return np.array(features_list), np.array(moves_list)


def simulate_localization(ship, L):
    """
    A simplified simulation of the localization process.
    Returns an estimate of moves needed to localize.
    """
    # moves roughly proportional to log(|L|)
    # with some randomness to simulate different layouts
    base_moves = np.log2(len(L)) * 2
    randomness = random.uniform(0.7, 1.3)
    
    # factor because bigger ships take more moves to get around
    ship_factor = ship.N / 10
    
    # Estimate moves needed
    moves = int(base_moves * randomness * ship_factor)
    return max(1, moves)  # to ensure moves are at least 1


def train_model(model, X_train, y_train, X_test, y_test, epochs=20):
    """
    Train the model using gradient descent.
    This follows the same pattern as in the MNIST tutorial.
    """
    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    
    # Define loss function (Mean Squared Error for regression)
    criterion = nn.MSELoss()
    
    # Define optimizer (SGD with momentum)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Lists to store loss values
    train_losses = []
    test_losses = []
    
    # Training loop
    for epoch in range(epochs):
        # Set model to training mode
        model.train()
        
        # Forward pass
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        
        # Backward and optimize
        optimizer.zero_grad()  # Zero the parameter gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        # Track training loss
        train_losses.append(loss.item())
        
        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test_tensor)
            test_loss = criterion(test_outputs, y_test_tensor)
            test_losses.append(test_loss.item())
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
    
    return train_losses, test_losses


def plot_results(model, X, y, train_losses, test_losses):
    """
    Plot training results and model predictions.
    """
    # Create a figure with 2 subplots
    plt.figure(figsize=(12, 5))
    
    # Plot 1: Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Testing Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Training and Testing Loss')
    plt.legend()
    
    # Plot 2: Size of L vs. Moves
    plt.subplot(1, 2, 2)
    
    # Get model predictions
    X_tensor = torch.tensor(X, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).numpy().flatten()
    
    # Scatter plot of actual values
    L_sizes = X[:, 0]  # First feature is normalized L size
    plt.scatter(L_sizes, y, alpha=0.5, label='Actual')
    
    # Plot a smoothed prediction line
    sorted_indices = np.argsort(L_sizes)
    plt.plot(L_sizes[sorted_indices], predictions[sorted_indices], 'r-', 
             linewidth=2, label='Model predictions')
    
    plt.xlabel('Size of L (normalized)')
    plt.ylabel('Number of Moves')
    plt.title('Relationship Between |L| and Moves Needed')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Parameters
    ship_size = 10
    num_samples = 300
    epochs = 20
    
    # 1. Construct dataset
    print("Generating dataset...")
    X, y = collect_data(num_samples, ship_size)
    
    # Split into training and testing sets (80/20)
    split_idx = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")
    
    # 2. Construct model
    print("Creating model...")
    input_size = X.shape[1]
    model = SimpleLocalizationModel(input_size)
    
    # 3 & 4. Define loss and train model
    print("Training model...")
    train_losses, test_losses = train_model(
        model, X_train, y_train, X_test, y_test, epochs
    )
    
    # 5. Evaluate model
    print("Evaluating model...")
    
    # Make predictions on test set
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy().flatten()
    
    # Calculate Mean Absolute Error
    mae = np.mean(np.abs(predictions - y_test))
    print(f"Mean Absolute Error: {mae:.2f} moves")
    
    # Plot results
    plot_results(model, X, y, train_losses, test_losses)

if __name__ == "__main__":
    main()