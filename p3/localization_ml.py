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

class LocalizationSoftmax(nn.Module):
    """
    Softmax classification model for predicting number of moves to localize.
    This follows the same structure as the MNIST softmax model in the tutorial.
    """
    def __init__(self, input_size, num_classes):
        super(LocalizationSoftmax, self).__init__()
        
        # Similar to the tutorial, we define a weight matrix and bias vector
        self.weight_matrix = torch.nn.Parameter(torch.randn(num_classes, input_size), requires_grad=True)
        self.bias_vector = torch.nn.Parameter(torch.randn(num_classes, 1), requires_grad=True)
    
    def forward(self, input_tensor):
        # Flatten input if it's not already
        flattened = nn.Flatten()(input_tensor)
        
        # Linear transformation: Wx + b
        linear_transformation = torch.matmul(self.weight_matrix, flattened.t()) + self.bias_vector
        
        # Transpose to get (batch_size, num_classes)
        logits = linear_transformation.t()
        
        # Apply softmax to get probabilities
        final_probabilities = nn.Softmax(dim=1)(logits)
        
        # Return both probabilities and logits (just like in the tutorial)
        return final_probabilities, logits


def generate_initial_L(ship, size):
    """
    Generate a random set L of possible locations of the given size.
    """
    if size > len(ship.open_cells):
        size = len(ship.open_cells)
    
    # Randomly select 'size' open cells
    selected_cells = random.sample(list(ship.open_cells.keys()), size)
    return {k: k for k in selected_cells}


def create_location_representation(ship, L):
    """
    Create a binary grid representation of the location set L.
    1 where the robot might be, 0 elsewhere.
    
    This is similar to how MNIST represents images as matrices.
    """
    # Create empty grid
    grid = np.zeros((ship.N, ship.N), dtype=np.float32)
    
    # Set 1 for each possible location
    for pos in L.keys():
        grid[pos[0], pos[1]] = 1.0
    
    return grid


def bin_move_counts(move_counts, num_bins=10):
    """
    Convert continuous move counts into discrete bins for classification.
    
    Returns:
        bins: array of bin edges
        binned_counts: move counts converted to bin indices (classes)
    """
    # Determine bin edges to roughly evenly distribute data
    max_moves = max(move_counts) + 1
    bins = np.linspace(0, max_moves, num_bins + 1)
    
    # Convert move counts to bin indices
    binned_counts = np.digitize(move_counts, bins) - 1
    
    # Clip to ensure all values are within the valid range
    binned_counts = np.clip(binned_counts, 0, num_bins - 1)
    
    return bins, binned_counts


def generate_training_data(ship_size=20, num_samples=1000):
    """
    Generate training data by running the localization strategy on
    different initial sets L of varying sizes.
    
    Returns:
        X: Dataset of ship grids representing location sets
        y: Move counts binned into classes
        bins: The bin edges used for classification
    """
    ship = Ship(ship_size)
    ship.place_entities()
    bot = Bot(ship)
    
    X = []
    move_counts = []
    
    print(f"Generating {num_samples} training samples...")
    
    for _ in range(num_samples):
        # Random size between 2 and max size
        L_size = random.randint(2, len(ship.open_cells))
        
        # Generate initial L set
        initial_L = generate_initial_L(ship, L_size)
        
        # Run localization strategy and get move count
        final_pos, move_count = bot.get_moves(initial_L)
        
        # Create grid representation
        grid_representation = create_location_representation(ship, initial_L)
        
        X.append(grid_representation)
        move_counts.append(move_count)
    
    # Convert move counts to classes
    bins, y = bin_move_counts(move_counts)
    
    print(f"Move count bins: {bins}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, bins


def confusion_matrix(model, x, y, num_classes):
    """
    Compute confusion matrix to evaluate classification performance.
    
    This is directly based on the confusion_matrix function in the tutorial.
    """
    identification_counts = np.zeros(shape=(num_classes, num_classes), dtype=np.int32)
    
    # Convert input to tensor
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    # Compute predictions
    probabilities, _ = model(x_tensor)
    predicted_classes = torch.argmax(probabilities, dim=1)
    
    n = len(x)
    
    for i in range(n):
        actual_class = y[i]
        predicted_class = predicted_classes[i].item()
        
        identification_counts[actual_class, predicted_class] += 1
    
    return identification_counts


def get_batch(x, y, batch_size):
    """
    Get a random batch of data for stochastic gradient descent.
    """
    n = len(x)
    batch_indices = random.sample([i for i in range(n)], k=batch_size)
    
    x_batch = [x[i] for i in batch_indices]
    y_batch = [y[i] for i in batch_indices]
    
    return x_batch, y_batch


def train_model(model, x_train, y_train, num_classes, batch_size=64, epochs=10, learning_rate=0.01):
    """
    Train the model using stochastic gradient descent.
    
    This follows the tutorial's approach of using batches and computing loss
    for cross-entropy classification.
    """
    # Convert y_train to tensor
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    # Store loss history
    losses = []
    
    # Set up optimizer (using Adam instead of basic SGD for better performance)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Define loss function (Cross Entropy, as in the tutorial)
    loss_function = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        batches = 0
        
        # Process in batches
        for batch in range(0, len(x_train), batch_size):
            # Check if we have enough data for a full batch
            if batch + batch_size > len(x_train):
                continue
                
            # Get batch
            x_batch, y_batch = x_train[batch:batch+batch_size], y_train[batch:batch+batch_size]
            
            # Convert to tensors
            x_batch_tensor = torch.tensor(x_batch, dtype=torch.float32)
            y_batch_tensor = torch.tensor(y_batch, dtype=torch.long)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            _, logits = model(x_batch_tensor)
            
            # Calculate loss (using the logits, as in the tutorial)
            loss = loss_function(logits, y_batch_tensor)
            
            # Backward pass
            loss.backward()
            
            # Optimize
            optimizer.step()
            
            # Update total loss
            total_loss += loss.item()
            batches += 1
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / batches if batches > 0 else 0
        losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
        
        # Compute and display confusion matrix every few epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            cm = confusion_matrix(model, x_train, y_train, num_classes)
            print("Confusion Matrix:")
            print(cm)
            
            # Calculate accuracy
            accuracy = np.trace(cm) / np.sum(cm)
            print(f"Training Accuracy: {accuracy:.4f}")
    
    return losses


def plot_loss(losses):
    """Plot the training loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.grid(True)
    plt.show()


def plot_confusion_matrix(cm, classes):
    """
    Plot the confusion matrix to visualize classification performance.
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.show()


def plot_example_grids(x, predicted_classes, actual_classes, bins, num_examples=5):
    """
    Plot example grids with their predicted and actual move count classes.
    """
    indices = random.sample(range(len(x)), num_examples)
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, num_examples, i + 1)
        plt.imshow(x[idx], cmap='Blues')
        plt.title(f"Pred: {bins[predicted_classes[idx]]:.1f}-{bins[predicted_classes[idx]+1]:.1f}\nTrue: {bins[actual_classes[idx]]:.1f}-{bins[actual_classes[idx]+1]:.1f}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def main():
    # Parameters
    ship_size = 15  # Smaller size for faster computation
    num_samples = 1000
    batch_size = 64
    epochs = 20
    learning_rate = 0.01
    num_classes = 10  # Number of bins for move counts
    
    # Generate data
    print("Generating training data...")
    start_time = time.time()
    X_data, y_data, bins = generate_training_data(ship_size=ship_size, num_samples=num_samples)
    print(f"Data generation completed in {time.time() - start_time:.2f} seconds")
    
    # Split data into training and testing sets (80/20 split)
    split_idx = int(0.8 * len(X_data))
    X_train, X_test = X_data[:split_idx], X_data[split_idx:]
    y_train, y_test = y_data[:split_idx], y_data[split_idx:]
    
    print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
    
    # Create the model
    input_size = ship_size * ship_size  # Flattened grid size
    model = LocalizationSoftmax(input_size, num_classes)
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    losses = train_model(
        model, X_train, y_train, num_classes,
        batch_size=batch_size, epochs=epochs, learning_rate=learning_rate
    )
    print(f"Model training completed in {time.time() - start_time:.2f} seconds")
    
    # Plot loss curve
    plot_loss(losses)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    cm = confusion_matrix(model, X_test, y_test, num_classes)
    
    # Calculate accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Plot confusion matrix
    class_labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    plot_confusion_matrix(cm, class_labels)
    
    # Get predictions for test set
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    probabilities, _ = model(x_test_tensor)
    predicted_classes = torch.argmax(probabilities, dim=1).numpy()
    
    # Plot example grids
    plot_example_grids(X_test, predicted_classes, y_test, bins)
    
    # Save the model
    torch.save(model.state_dict(), 'localization_softmax_model.pth')
    print("Model saved as 'localization_softmax_model.pth'")


if __name__ == "__main__":
    main()