import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Define constants
IMG_WIDTH, IMG_HEIGHT = 48, 48  # FER-2013 image dimensions
BATCH_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define paths (update these to your actual paths)
train_dir = r"C:\Users\chinm\Downloads\emotion_recog_cleaned\train"
test_dir = r"C:\Users\chinm\Downloads\emotion_recog_cleaned\test"

# Emotion labels in FER-2013
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define data transformations
train_transforms = transforms.Compose([
    transforms.Grayscale(),  # Ensure grayscale
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),  # Convert to tensor and scale to [0,1]
    transforms.Normalize([0.5], [0.5])  # Normalize with mean and std for grayscale
])

test_transforms = transforms.Compose([
    transforms.Grayscale(),  # Ensure grayscale
    transforms.Resize((IMG_WIDTH, IMG_HEIGHT)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Load datasets
try:
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transforms)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # Print dataset information
    print(f"Using device: {DEVICE}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Class indices: {train_dataset.class_to_idx}")

except Exception as e:
    print(f"Error loading datasets: {e}")


# Visualize some training images
def plot_sample_images(dataloader, n=12):
    plt.figure(figsize=(12, 8))

    # Get batch of images
    data_iter = iter(dataloader)
    images, labels = next(data_iter)

    for i in range(min(n, len(images))):
        plt.subplot(3, 4, i + 1)
        # Convert from tensor to numpy for display
        img = images[i].squeeze().numpy()  # Remove channel dim and convert to numpy
        plt.imshow(img, cmap='gray')

        # Get the class name
        class_idx = labels[i].item()
        plt.title(emotion_labels[class_idx])
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# Visualize sample images
plot_sample_images(train_loader)


# Check class distribution
def check_class_distribution(dataset, labels):
    counts = [0] * len(labels)
    for _, label in dataset:
        counts[label] += 1

    plt.figure(figsize=(10, 6))
    plt.bar(labels, counts)
    plt.title('Class Distribution')
    plt.xlabel('Emotion Classes')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Print counts
    for label, count in zip(labels, counts):
        print(f"{label}: {count} samples")

# Uncomment to check class distribution
check_class_distribution(train_dataset, emotion_labels)

# Visualize sample images from test set
print("\nSample test images:")
plot_sample_images(test_loader)

# Check class distribution for test set
print("\nClass distribution in test dataset:")
check_class_distribution(test_dataset, emotion_labels)
