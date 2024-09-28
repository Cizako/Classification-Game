import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision
import matplotlib.pyplot as plt

class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        print("loading dataset")
        # Traverse the directory structure to collect image paths and labels
        for class_idx, class_name in enumerate(os.listdir(root_dir)):
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

        # Load image
        image = Image.open(img_path).convert("RGB")

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)

        return image, label
    

    def visualize(self, nr_images=5):
        """
        Visualize a specified number of images from the dataset with their true labels.
        
        Args:
        - nr_images (int): Number of images to visualize.
        """
        # Randomly select 'nr_images' indices

       
        images = []
        labels = []

        # Load images and labels
        for idx in  range(nr_images):
            image, label = self.__getitem__(idx)
            images.append(image)
            labels.append(label)

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

if __name__ == "__main__":
    data = CustomImageDataset('data/raw-img')

    data.visualize(1)