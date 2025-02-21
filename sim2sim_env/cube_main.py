import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
import pybullet as p
import numpy as np
import torch.nn.functional as F
from sim2sim_env import Sim2SimEnv
from cube_dataset import CubeDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from vgg16 import CubeDetectorVGG16
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.utils as utils

def collect_data(num_samples=1000, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    image_dir = os.path.join(data_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    labels = []
    
    env = Sim2SimEnv(renders=False, cameraRandom=1)
    
    for i in tqdm(range(num_samples)):
        obs = env.reset()
        # Get image from environment
        light_direction, light_color, light_distance, light_specular = env.randomize_lighting()

        img_arr = p.getCameraImage(width=env._width,
                                height=env._height,
                                viewMatrix=env._view_matrix,
                                projectionMatrix=env._proj_matrix,
                                lightDirection=light_direction,
                                lightColor=light_color,
                                lightDistance=light_distance,
                                shadow=1,
                                renderer=p.ER_TINY_RENDERER)
        rgb_image = img_arr[2]
        rgb_array = np.array(rgb_image).reshape(48, 48, 4)[:, :, :3].astype(np.uint8)

        # Now convert the numpy array to an image using PIL
        image = Image.fromarray(rgb_array)

        # Save image
        image_path = os.path.join(image_dir, f"image_{i}.png")
        image.save(image_path)
        
        # Get cube position
        cube_position, cube_orientation = env.get_cube_position_relative_to_camera()
        
        # Save label (x, y, z coordinates and orientation)
        labels.append({"image_path": image_path, "position": cube_position, "orientation": cube_orientation})

    # Save labels to a file
    with open(os.path.join(data_dir, "labels.txt"), "w") as f:
        for label in labels:
            f.write(f"{label['image_path']} {label['position'][0]} {label['position'][1]} {label['position'][2]} {label['orientation'][0]} {label['orientation'][1]} {label['orientation'][2]} {label['orientation'][3]}\n")
    
    
def quaternion_distance_loss(pred_orientation, target_orientation):
    """
    Computes the quaternion distance loss:
    L_quat = 1 - |dot(pred, target)|
    """
    # Normalize quaternions to ensure they are unit quaternions
    pred_orientation = F.normalize(pred_orientation, dim=1)
    target_orientation = F.normalize(target_orientation, dim=1)
    
    # Compute the absolute dot product between predicted and target quaternions
    dot_product = torch.sum(pred_orientation * target_orientation, dim=1)
    return torch.mean(1 - torch.abs(dot_product))

def combined_loss(prediction, target, pos_weight=1.0, ori_weight=1.0):
    """
    Combines position and orientation loss with adjustable weights.
    """
    # Split predictions and targets into position and orientation parts
    pred_position, pred_orientation = prediction[:, :3], prediction[:, 3:]
    target_position, target_orientation = target[:, :3], target[:, 3:]

    # Position loss
    position_loss = F.mse_loss(pred_position, target_position)

    # Orientation loss (quaternion distance loss)
    orientation_loss = quaternion_distance_loss(pred_orientation, target_orientation)

    # Combine losses with weights
    total_loss = pos_weight * position_loss + ori_weight * orientation_loss
    return total_loss

def validate_model(device, val_loader, model, criterion):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in val_loader:
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
    return val_loss / len(val_loader)

# Training loop
def train_detector(device, data_loader, val_loader, model, criterion, optimizer, epochs=20, patience=5, clip_value=1.0):
    model.train()
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # Initialize learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    for epoch in range(epochs):
        running_loss = 0.0

        # Training Loop
        for images, targets in data_loader:
            images, targets = images.to(device), targets.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)

            # Backward pass
            loss.backward()

            # Apply gradient clipping
            utils.clip_grad_norm_(model.parameters(), max_norm=clip_value)

            # Optimizer step
            optimizer.step()

            running_loss += loss.item()

        # Validate the model
        val_loss = validate_model(device, val_loader, model, criterion)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(data_loader):.4f}, Val Loss: {val_loss:.4f}")

        # Step the scheduler based on validation loss
        scheduler.step(val_loss)

        # Check for improvement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "best_model.pt")  # Save the best model
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered!")
            break
        
def train(batch_size=25, lr=1e-4, epochs=20, patience=5):
    # Define image transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resizing for VGG16
        transforms.ToTensor(),
    ])

    # Create the dataset and split into train and validation
    dataset = CubeDataset(data_dir="cube_data_dist", transform=transform)
    val_split = 0.2
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Instantiate the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CubeDetectorVGG16(pretrained=True).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)
    elif torch.cuda.device_count() == 1:
        print("Using 1 GPU")
    else:
        print("Using CPU")  

    # Loss and optimizer
    criterion = combined_loss
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

    # Train the model
    train_detector(device, train_loader, val_loader, model, criterion, optimizer, epochs=epochs, patience=patience)

    # Save the final model
    torch.save(model.state_dict(), f"cube_detector_vgg16_{batch_size}_{int(lr*1e4)}_dist.pt")

def main():
    collect_data(num_samples=10000, data_dir="cube_data_dist")
    train(batch_size=25, lr=2e-4, epochs=30)

if __name__ == "__main__":
    main()