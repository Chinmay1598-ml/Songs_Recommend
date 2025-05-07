import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pickle

# Import from our previous files
from environment_data_load import test_loader, DEVICE, emotion_labels
from model_architecture import EmotionCNN
from model_training import evaluate_model

# Constants
NUM_CLASSES = 7
MODEL_PATH = 'emotion_model_best.pth'


# Function to get all predictions
def get_predictions(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def evaluate_in_detail():
    # Create and load model
    model = EmotionCNN(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))

    # Loss function for evaluation
    criterion = nn.CrossEntropyLoss()

    # Get test loss and accuracy
    test_loss, test_acc = evaluate_model(model, test_loader, criterion)
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Get predictions for confusion matrix
    y_pred, y_true = get_predictions(model, test_loader)

    # Plot confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=emotion_labels,
        yticklabels=emotion_labels
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Print classification report
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=emotion_labels, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=emotion_labels))

    # Plot precision, recall, and f1-score for each class
    classes = list(report.keys())[:-3]  # Exclude accuracy, macro avg, weighted avg
    metrics = ['precision', 'recall', 'f1-score']

    plt.figure(figsize=(15, 5))
    x = np.arange(len(classes))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in classes]
        plt.bar(x + i * width, values, width, label=metric)

    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Precision, Recall, and F1-score by Class')
    plt.xticks(x + width, classes, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('classification_metrics.png')
    plt.show()

    # Compare with training history if available
    try:
        with open('training_history.pkl', 'rb') as f:
            history = pickle.load(f)

        # Plot final training vs validation
        plt.figure(figsize=(12, 5))

        # Plot accuracies
        plt.subplot(1, 2, 1)
        plt.plot(history['train_acc'], 'b-', label='Training')
        plt.plot(history['val_acc'], 'r-', label='Validation')
        plt.axhline(y=test_acc, color='g', linestyle='--', label=f'Test ({test_acc:.4f})')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        # Plot losses
        plt.subplot(1, 2, 2)
        plt.plot(history['train_loss'], 'b-', label='Training')
        plt.plot(history['val_loss'], 'r-', label='Validation')
        plt.axhline(y=test_loss, color='g', linestyle='--', label=f'Test ({test_loss:.4f})')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.tight_layout()
        plt.savefig('final_comparison.png')
        plt.show()
    except Exception as e:
        print(f"Could not load training history: {e}")


if __name__ == "__main__":
    evaluate_in_detail()