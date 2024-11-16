import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample by 2
        )

        # Middle layer (no downsampling here)
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder (includes upsampling)
        self.upsample = nn.ConvTranspose2d(128, 128, kernel_size=2, stride=2)  # Upsample by 2
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_channels, kernel_size=1)  # Final output layer
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder(x)

        # Middle layer
        x2 = self.middle(x1)

        # Decoder
        x3 = self.upsample(x2)  # Upsample the feature map
        x4 = self.decoder(x3)   # Apply final convolutions

        return x4
