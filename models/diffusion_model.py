import torch
import torch.nn as nn
from models.unet import UNet

class DiffusionModel(nn.Module):
    def __init__(self, unet_model, timesteps=1000):
        super(DiffusionModel, self).__init__()
        
        # U-Net model for predicting depth map
        self.unet = unet_model
        self.timesteps = timesteps

    def forward(self, x, t):
        # Forward pass through U-Net
        return self.unet(x)
    
    def q_sample(self, x, t):
        """
        Add noise to the input `x` at timestep `t`.
        This is a simplified noise schedule. You can adjust the schedule if needed.
        """
        noise = torch.randn_like(x)  # Random noise
        return x + noise  # Add noise to the input

    def compute_loss(self, predicted_noise, target_depth):
        """
        Compute loss between predicted noise and true depth.
        This uses the mean squared error between the predicted noise and the target depth image.
        """
        loss = nn.MSELoss()(predicted_noise, target_depth)
        return loss
