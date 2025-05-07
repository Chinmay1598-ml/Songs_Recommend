import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# Import from our previous files
from environment_data_load import train_loader, test_loader, DEVICE, emotion_labels
from model_architecture import EmotionCNN

# Constants
NUM_CLASSES = 7
EPOCHS = 30
MODEL_SAVE_PATH = 'emotion_model_best.pth'

# Create model instance
model = EmotionCNN(NUM_CLASSES).to(DEVICE)
print(model)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)


# Function to evaluate the model
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Calculate metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


# Training function
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=EPOCHS):
    # Lists to track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    # For early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 10

    # To track training time
    start_time = time.time()

    # Loop over epochs
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        # Validation phase
        val_loss, val_acc = evaluate_model(model, test_loader, criterion)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Update learning rate based on validation loss
        scheduler.step(val_loss)

        # Print epoch statistics
        print(f'Epoch {epoch + 1}/{num_epochs}, '
              f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved best model with val_loss: {val_loss:.4f}")
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    # Calculate training time
    training_time = time.time() - start_time
    print(f"Training completed in {training_time / 60:.2f} minutes")

    # Load best model weights
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))

    return model, history


# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


if __name__ == "__main__":
    # Train the model
    model, history = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, EPOCHS)

    # Plot the training history
    plot_training_history(history)

    # Save the history for later analysis
    import pickle

    with open('training_history.pkl', 'wb') as f:
        pickle.dump(history, f)

    print(f"Best model saved to {MODEL_SAVE_PATH}")
    print("Training history saved to training_history.pkl")