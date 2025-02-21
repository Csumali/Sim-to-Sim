import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from torchvision.utils import save_image

# Dataset class for loading images
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


# Define the generator (U-Net-like structure)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 48 -> 24
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 24 -> 12
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 24 -> 48
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Define the discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=4, stride=2, padding=1),  # 128 -> 64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64 -> 32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 32 * 32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)

def gradient_penalty(discriminator, real_data, fake_data):
    alpha = torch.rand(real_data.size(0), 1, 1, 1).to(device)
    interpolates = alpha * real_data + (1 - alpha) * fake_data
    interpolates.requires_grad_(True)
    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()

# Training function for GraspGAN
def train_graspgan(generator, discriminator, optimizer_g, optimizer_d, pybullet_loader, maniskill_loader, epochs=10):
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for (pybullet_images, maniskill_images) in zip(pybullet_loader, maniskill_loader):
            # Send data to GPU if available
            pybullet_images = pybullet_images.to(device)
            maniskill_images = maniskill_images.to(device)

            # Generate adapted images
            adapted_images = generator(pybullet_images)

            # Create discriminator input
            adapted_images_resized = nn.functional.interpolate(adapted_images, size=(128, 128), mode="bilinear")
            fake_input = torch.cat((adapted_images_resized, maniskill_images), dim=1)

            # Train discriminator
            for _ in range(2):
                optimizer_d.zero_grad()
                real_labels = torch.ones(fake_input.size(0), 1).to(device) * 0.9
                fake_labels = torch.zeros(fake_input.size(0), 1).to(device) * 0.1
                real_loss = criterion(discriminator(torch.cat((maniskill_images, maniskill_images), dim=1)), real_labels)
                fake_loss = criterion(discriminator(fake_input.detach()), fake_labels)
                d_loss = real_loss + fake_loss
                # gp = gradient_penalty(discriminator, real_labels, fake_labels)
                # d_loss += 10 * gp  # Weight the penalty appropriately
                d_loss.backward()
                optimizer_d.step()

            # Train generator
            optimizer_g.zero_grad()
            # Generator loss needs the discriminator output
            g_loss = criterion(discriminator(fake_input), real_labels)
            g_loss.backward()
            optimizer_g.step()
        save_image(adapted_images, f"generated_epoch_{epoch}.png", normalize=True)

        print(f"Epoch [{epoch+1}/{epochs}], Generator Loss: {g_loss.item()}, Discriminator Loss: {d_loss.item()}")


if __name__ == "__main__":
    # Set up paths
    pybullet_image_paths = [os.path.join("cube_data/images", f) for f in os.listdir("cube_data/images") if f.endswith(".png")]
    maniskill_image_paths = [os.path.join("../maniskill/data", f) for f in os.listdir("../maniskill/data") if f.endswith(".png")]

    # Transforms for both datasets
    pybullet_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.ToTensor(),
    ])

    maniskill_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Datasets and DataLoaders
    pybullet_dataset = ImageDataset(pybullet_image_paths, pybullet_transform)
    maniskill_dataset = ImageDataset(maniskill_image_paths, maniskill_transform)

    print("Loading Data..")
    pybullet_loader = DataLoader(pybullet_dataset, batch_size=16, shuffle=True)
    maniskill_loader = DataLoader(maniskill_dataset, batch_size=16, shuffle=True)

    # Initialize models and optimizers
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-6)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=1e-4)

    # Train GraspGAN
    print("Training..")
    train_graspgan(generator, discriminator, optimizer_g, optimizer_d, pybullet_loader, maniskill_loader, epochs=30)
    
    # Define paths for saving
    model_save_path = "gan_models"
    image_save_path = "generated_images"

    # Create directories if they don't exist
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)

    # Save the trained generator model
    torch.save(generator.state_dict(), os.path.join(model_save_path, "graspgan_generator.pth"))
    print(f"Generator model saved at {os.path.join(model_save_path, 'graspgan_generator.pth')}")

    # Save a few generated images
    generator.eval()  # Switch to evaluation mode
    with torch.no_grad():
        for batch_idx, pybullet_images in enumerate(pybullet_loader):
            pybullet_images = pybullet_images.to(device)
            adapted_images = generator(pybullet_images)  # Generate adapted images
            
            # Save images from this batch
            for i, img in enumerate(adapted_images):
                save_image(img, os.path.join(image_save_path, f"adapted_image_{batch_idx * len(adapted_images) + i}.png"))
            
            if batch_idx >= 4:  # Save images from the first 5 batches only (adjust as needed)
                break

    print(f"Generated images saved in {image_save_path}")