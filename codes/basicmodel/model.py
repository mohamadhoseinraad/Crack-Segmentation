import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomCrackSegModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(CustomCrackSegModel, self).__init__()

        # Encoder
        self.enc1_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc2_conv = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)

        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc3_conv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.enc4_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # nn.Conv2d(512, 512, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        e1 = self.enc1_conv(x)  # 512 512 32
        p1 = self.pool1(e1)  # 256 256 32

        e2 = self.enc2_conv(p1)  # 256 256 64
        p2 = self.pool2(e2)  # 128 128 64

        e3 = self.enc3_conv(p2)  # 128 128 128
        p3 = self.pool3(e3)  # 64 64 128

        e4 = self.enc4_conv(p3)  # 64 64 256
        p4 = self.pool4(e4)  # 32 32 256

        # Bottleneck
        b = self.bottleneck_conv(p4)  # 32 32 512

        # Decoder path with skip connections
        d4 = self.upconv4(b) # 64 64 256
        d4 = torch.cat((d4, e4), dim=1) # 64 64 512
        d4 = self.dec4_conv(d4) # 64 64 256

        d3 = self.upconv3(d4)
        d3 = torch.cat((d3, e3), dim=1)
        d3 = self.dec3_conv(d3)

        d2 = self.upconv2(d3)
        d2 = torch.cat((d2, e2), dim=1)
        d2 = self.dec2_conv(d2)

        d1 = self.upconv1(d2)
        d1 = torch.cat((d1, e1), dim=1)
        d1 = self.dec1_conv(d1)

        output = self.out_conv(d1)
        return output


if __name__ == "__main__":
    
    model = CustomCrackSegModel(3, 1)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")


    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")