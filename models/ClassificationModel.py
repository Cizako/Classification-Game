import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):
    def __init__(self, num_classes=10):
        super(ClassificationModel, self).__init__()
        
        # First convolutional layer: 3 input channels (RGB), 32 output channels, kernel size 5, padding 2 to preserve size
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2)
        
        # Second convolutional layer: outputs a 32-channel feature map
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        
        # Third convolutional layer: further reduces spatial dimensions
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, padding=2)

        # Fully connected layer
        self.fc1 = nn.Linear(128 * 62 * 62, 512)  # Adjusted for the final size after pooling
        
        # Prediction layer
        self.prediction = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # Input size is assumed to be (batch_size, 3, 500, 500)
        
        # First conv -> ReLU -> Max Pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)  # Output: (batch_size, 32, 250, 250)

        # Second conv -> ReLU -> Max Pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)  # Output: (batch_size, 64, 125, 125)

        # Third conv -> ReLU -> Max Pooling
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, kernel_size=2)  # Output: (batch_size, 128, 62, 62)

        # Flatten the tensor for fully connected layer
        x = x.view(x.size(0), -1)  # Output: (batch_size, 128 * 62 * 62)

        # Fully connected layer -> ReLU
        x = F.relu(self.fc1(x))

        # Output layer (no activation, to be combined with a loss function later)
        x = self.prediction(x)

        return x