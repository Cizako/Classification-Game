import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt
import random
from translate import translate as translation
import torch

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, amount_of_classes = 2):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        self.nr_classes = 0

        # Traverse the directory structure to collect image paths and labels
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
            if self.nr_classes >= amount_of_classes:
                break
            self.nr_classes += 1
            
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                self.class_to_idx[class_name] = class_idx
                for img_name in os.listdir(class_dir):
                    img_path = os.path.join(class_dir, img_name)
                    if img_path.endswith(".jpg") or img_path.endswith(".jpeg"):  # Handle only JPEGs
                        self.image_paths.append(img_path)
                        self.labels.append(class_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        target = torch.zeros(self.nr_classes)
        target[label] = 1

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, target
    
    def visualize(self, nr_images=5):
        """
        Visualize a specified number of images from the dataset with their true labels.
        
        Args:
        - nr_images (int): Number of images to visualize.
        """
        images = []
        labels = []

        # Randomly select 'nr_images' indices
        indices = random.sample(range(len(self.image_paths)), nr_images)

        for idx in indices:
            image, target = self.__getitem__(idx)
            images.append(image)
            
            # Convert the one-hot target tensor back to class index
            label_idx = torch.argmax(target).item()
            labels.append(label_idx)

        # Create a grid of images
        grid_img = torchvision.utils.make_grid(images, nrow=nr_images, normalize=True)

        # Plot the images
        plt.figure(figsize=(12, 6))
        plt.imshow(grid_img.permute(1, 2, 0))  # permute to (height, width, channels)
        plt.axis('off')

        # Add titles below the images showing the true labels
        label_names = [list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(lbl)] for lbl in labels]
        plt.title(" | ".join(label_names))  # Titles are class names for the images
        
        plt.show()

    def visualize_all_classes(self):
        """
        Visualize one random image from each class in the dataset.
        """
        images = []
        labels = []

        # Get one random image from each class
        for class_name, class_idx in self.class_to_idx.items():
            class_images = [i for i, label in enumerate(self.labels) if label == class_idx]
            if class_images:
                random_idx = random.choice(class_images)
                image, target = self.__getitem__(random_idx)
                images.append(image)

                # Convert the one-hot target tensor back to class index
                label_idx = torch.argmax(target).item()
                labels.append(label_idx)

        # Create a grid of images
        grid_img = torchvision.utils.make_grid(images, nrow=len(images), normalize=True)

        # Plot the images
        plt.figure(figsize=(16, 4))
        plt.imshow(grid_img.permute(1, 2, 0))  # permute to (height, width, channels)
        plt.axis('off')

        # Add titles below the images showing the class names
        label_names = [list(self.class_to_idx.keys())[list(self.class_to_idx.values()).index(lbl)] for lbl in labels]
        plt.title(" | ".join(label_names))  # Titles are class names for the images
        
        plt.show()


if __name__ == "__main__":

    transform = transforms.Compose([
    transforms.Resize((500, 500)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

    data = CustomImageDataset('data/raw-img', transform=transform, amount_of_classes=5 )

    data.visualize(10)

    print(data.nr_classes)