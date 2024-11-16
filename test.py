import torch
import os
from torch.utils.data import Dataset, DataLoader
from models.unet import UNet  # Assuming you have a simplified version of UNet
from models.diffusion_model import DiffusionModel  # Assuming diffusion model uses default parameters
from torchvision import transforms
from PIL import Image
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Custom Dataset Class to load all relevant images (depth, intensity, etc.)
class DepthDataset(Dataset):
    def __init__(self, nir_dir, thr_dir, transform=None, target_size=(128, 128), depth_type="depth"):
        self.nir_dir = nir_dir
        self.thr_dir = thr_dir
        self.transform = transform
        self.target_size = target_size
        self.depth_type = depth_type

        # Recursively get image paths
        self.nir_paths = self.get_all_image_paths(nir_dir)
        self.thr_paths = self.get_all_image_paths(thr_dir)

    def get_all_image_paths(self, directory):
        image_paths = []
        for root, _, files in os.walk(directory):  # os.walk to search recursively
            for filename in files:
                if filename.endswith('.png'):
                    image_paths.append(os.path.join(root, filename))
        return image_paths

    def __len__(self):
        return len(self.nir_paths)

    def __getitem__(self, idx):
        # Load the NIR and Thermal images
        nir_img = Image.open(self.nir_paths[idx]).convert("L")  # Single channel for NIR
        thr_img = Image.open(self.thr_paths[idx]).convert("L")  # Single channel for THR

        # Resize images to target size
        nir_img = nir_img.resize(self.target_size)
        thr_img = thr_img.resize(self.target_size)

        # Convert images to numpy arrays
        nir_img = np.array(nir_img, dtype=np.uint8)
        thr_img = np.array(thr_img, dtype=np.uint8)

        # Stack NIR and THR as single channels
        combined_input = np.dstack((nir_img, thr_img))

        # Convert to PIL Image for transformations
        combined_input = Image.fromarray(combined_input)

        # Apply transformations
        if self.transform:
            combined_input = self.transform(combined_input)

        # Ensure the transformed input is float32 for the model
        combined_input = combined_input.float()

        # Load depth image
        depth_img_path = self.nir_paths[idx]
        depth_img = Image.open(depth_img_path).convert("L")  # Single channel for depth
        depth_img = depth_img.resize(self.target_size)
        depth_img = np.array(depth_img)
        depth_img = transforms.ToTensor()(depth_img).float()

        # Normalize depth
        depth_img = transforms.Normalize(mean=[0.5], std=[0.5])(depth_img)

        return combined_input, depth_img


# Testing function
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([ 
        transforms.ToTensor(),
        transforms.Resize((128, 128)),  # Resizing images to fixed size (if needed)
        transforms.Normalize(mean=[0.5, 0.5], std=[0.5, 0.5]),  # Normalizing for the input
    ])

    # Paths for the NIR and Thermal images (Set your actual paths here)
    nir_path = "C:/Users/mrpre/Documents/PPRS_PoC/data/nir"
    thr_path = "C:/Users/mrpre/Documents/PPRS_PoC/data/thr"

    # Create dataset and dataloader
    dataset = DepthDataset(nir_path, thr_path, transform=transform)
    test_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Initialize U-Net model with simplified architecture
    unet_model = UNet(in_channels=2, out_channels=1)  # 2 channels for NIR + Thermal input, 1 for depth output
    model = DiffusionModel(unet_model)
    
    # Load the saved model
    model_path = "C:/Users/mrpre/Documents/PPRS_PoC/models/depth_estimation_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    all_true_depths = []
    all_pred_depths = []

    # Loop through the test dataset
    with torch.no_grad():
        for i, (inputs, depths) in enumerate(test_loader):
            inputs, depths = inputs.to(device), depths.to(device)

            # Random timestep t for each image in the batch
            t = torch.randint(0, model.timesteps, (inputs.size(0),), device=device).long()

            # Add noise to inputs using the `q_sample` method
            noisy_inputs = model.q_sample(inputs, t)

            # Forward pass through diffusion model
            predicted_noise = model(noisy_inputs, t)

            # Collect predictions and true depths
            all_true_depths.extend(depths.cpu().numpy())
            all_pred_depths.extend(predicted_noise.detach().cpu().numpy())

            # Visualize and save predicted depth maps
            for j in range(inputs.size(0)):
                true_depth = depths[j].cpu().numpy().squeeze()
                pred_depth = predicted_noise[j].cpu().numpy().squeeze()

                # Plot true and predicted depth maps side-by-side
                fig, axs = plt.subplots(1, 2, figsize=(8, 4))
                axs[0].imshow(true_depth, cmap='gray')
                axs[0].set_title("True Depth")
                axs[1].imshow(pred_depth, cmap='gray')
                axs[1].set_title("Predicted Depth")

                # Save the image
                plt.savefig(f"predicted_depth_{i * 4 + j}.png")
                plt.close(fig)

                # Display first few depth maps if desired
                if i < 2 and j < 2:
                    plt.show()

    # Flatten the lists to compute MSE
    all_true_depths = np.array(all_true_depths).flatten()
    all_pred_depths = np.array(all_pred_depths).flatten()

    # Calculate MSE
    mse = mean_squared_error(all_true_depths, all_pred_depths)
    print(f"Mean Squared Error (MSE) on Test Data: {mse:.4f}")


if __name__ == "__main__":
    test()
