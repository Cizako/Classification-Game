from Dataset import CustomImageDataset
from torchvision import transforms, utils

amount_of_classes = 2

transform = transforms.Compose([
    transforms.Resize((500, 500)), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = CustomImageDataset('data/raw-img', transform)

print(len(dataset.labels))
print(dataset.class_to_idx)