import torch
import torch.nn as nn

class UNet(nn.Module):
    """U-Net model for binary segmentation of road skeletons."""
    def __init__(self, in_channels=1, out_channels=1, init_features=64):
        super(UNet, self).__init__()
        features = init_features
        # Encoder (downsampling)
        self.encoder1 = UNet._block(in_channels, features)
        self.pool1   = nn.MaxPool2d(2, 2)
        self.encoder2 = UNet._block(features, features*2)
        self.pool2   = nn.MaxPool2d(2, 2)
        self.encoder3 = UNet._block(features*2, features*4)
        self.pool3   = nn.MaxPool2d(2, 2)
        self.encoder4 = UNet._block(features*4, features*8)
        self.pool4   = nn.MaxPool2d(2, 2)
        # Bottleneck
        self.bottleneck = UNet._block(features*8, features*16)
        # Decoder (upsampling)
        self.upconv4  = nn.ConvTranspose2d(features*16, features*8, kernel_size=2, stride=2)
        self.decoder4 = UNet._block(features*8 * 2, features*8)   # concat, so *2 channels
        self.upconv3  = nn.ConvTranspose2d(features*8, features*4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block(features*4 * 2, features*4)
        self.upconv2  = nn.ConvTranspose2d(features*4, features*2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block(features*2 * 2, features*2)
        self.upconv1  = nn.ConvTranspose2d(features*2, features, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(features * 2, features)
        # Final 1x1 conv
        self.conv_last = nn.Conv2d(features, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        bott = self.bottleneck(self.pool4(enc4))
        # Decoder path with skip connections
        dec4 = self.upconv4(bott)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        # Output logits
        out = self.conv_last(dec1)
        return out

    @staticmethod
    def _block(in_channels, features):
        # Two convolutional layers with ReLU (and BatchNorm)
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )
