import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class CubeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        
        # Load data from file
        with open(os.path.join(data_dir, "labels.txt"), "r") as file:
            for line in file:
                parts = line.strip().split()
                image_path = parts[0]
                # Convert position and orientation to float and store as a single tensor
                position_orientation = list(map(float, parts[1:]))  # Get all 7 elements
                self.data.append((image_path, torch.tensor(position_orientation)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, target = self.data[idx]
        
        # Open and preprocess the image
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            
        return image, target
