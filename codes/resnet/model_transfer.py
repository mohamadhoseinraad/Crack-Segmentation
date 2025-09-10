import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet34_Weights

class PreTrainedEncoderCrackSegModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, dropout_p=0.3, pretrained=True):
        super(PreTrainedEncoderCrackSegModel, self).__init__()
        
        
        resnet = models.resnet34(weights=ResNet34_Weights.DEFAULT) if pretrained else models.resnet34(weights=None)
        
        
        self.encoder1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        
        self.bridge_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        
        self.upconv5 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p)
        )
        self.dec5_conv = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3_conv = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2_conv = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Final upconv removed as we'll use interpolation for more flexibility
        self.dec1_conv = nn.Sequential(
            nn.Conv2d(96, 32, kernel_size=3, padding=1),  # Changed input from 64 to 96 (32+64)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_p * 0.5),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)
        
        
        if pretrained:
            self._freeze_encoder()

    def _freeze_encoder(self):
        
        for param in self.encoder1.parameters():
            param.requires_grad = False
        for param in self.encoder2.parameters():
            param.requires_grad = False
        for param in self.encoder3.parameters():
            param.requires_grad = False
        for param in self.encoder4.parameters():
            param.requires_grad = False

    def forward(self, x):
        
        input_size = x.shape[2:]
        
        # Encoder ResNet
        e1 = self.encoder1(x)       # [B, 64, H/2, W/2]
        e2 = self.encoder2(e1)      # [B, 64, H/2, W/2]
        e3 = self.encoder3(e2)      # [B, 128, H/4, W/4]
        e4 = self.encoder4(e3)      # [B, 256, H/8, W/8]
        e5 = self.encoder5(e4)      # [B, 512, H/16, W/16]
        
        # Bridge
        b = self.bridge_conv(e5)    # [B, 1024, H/16, W/16]

        # Decoder
        
        d5 = self.upconv5(b)                                # [B, 512, H/16, W/16]
        if d5.shape[2:] != e5.shape[2:]:
            d5 = F.interpolate(d5, size=e5.shape[2:], mode='bilinear', align_corners=False)
        d5 = torch.cat((d5, e5), dim=1)                    # [B, 1024, H/16, W/16]
        d5 = self.dec5_conv(d5)                            # [B, 512, H/16, W/16]

        d4 = self.upconv4(d5)                              # [B, 256, H/8, W/8]
        if d4.shape[2:] != e4.shape[2:]:
            d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat((d4, e4), dim=1)                    # [B, 512, H/8, W/8]
        d4 = self.dec4_conv(d4)                            # [B, 256, H/8, W/8]

        d3 = self.upconv3(d4)                              # [B, 128, H/4, W/4]
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat((d3, e3), dim=1)                    # [B, 256, H/4, W/4]
        d3 = self.dec3_conv(d3)                            # [B, 128, H/4, W/4]

        d2 = self.upconv2(d3)                              # [B, 64, H/2, W/2]
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat((d2, e2), dim=1)                    # [B, 128, H/2, W/2]
        d2 = self.dec2_conv(d2)                            # [B, 64, H/2, W/2]

        d1 = self.upconv1(d2)                              # [B, 32, H, W]
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat((d1, e1), dim=1)                    # [B, 96, H/2, W/2]
        d1 = self.dec1_conv(d1)                            # [B, 32, H/2, W/2]

        # Final output with resize to original input size
        output = self.out_conv(d1)                         # [B, 1, H/2, W/2]
        if output.shape[2:] != input_size:
            output = F.interpolate(output, size=input_size, mode='bilinear', align_corners=False)

        return output

if __name__ == "__main__":

    print("\n=== Pre-trained Encoder Model ===")
    pretrained_model = PreTrainedEncoderCrackSegModel(3, 1, dropout_p=0.3, pretrained=True)
    pretrained_params = sum(p.numel() for p in pretrained_model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {pretrained_params:,}")
    
    # check dims
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = pretrained_model(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
