import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, input_channels=3):
        super(UNet, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.act = nn.LeakyReLU(0.1, inplace=True)

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )

        self.enc0 = conv_block(input_channels, 48)
        self.enc1 = conv_block(48, 48)
        self.enc2 = conv_block(48, 48)
        self.enc3 = conv_block(48, 48)
        self.enc4 = conv_block(48, 48)
        self.enc5 = nn.Sequential(
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(48, 48, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        def dec_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.LeakyReLU(0.1, inplace=True)
            )

        # self.dec5 = dec_block(96, 96)
        # self.dec4 = dec_block(96, 96)
        # self.dec3 = dec_block(96, 96)
        # self.dec2 = dec_block(96, 96)
        # self.dec1 = nn.Sequential(
        #     nn.Conv2d(96, 64, 3, padding=1),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Conv2d(64, 32, 3, padding=1),
        #     nn.LeakyReLU(0.1, inplace=True)
        # )

        
        self.dec5 = dec_block(48 + 48, 96)                        
        self.dec4 = dec_block(96 + 48, 96)       
        self.dec3 = dec_block(96 + 48, 96)        
        self.dec2 = dec_block(96 + 48, 96)        
        
        self.dec1 = nn.Sequential(
            nn.Conv2d(96 + 48, 64, 3, padding=1), # 144 ➜ 64
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.final_conv = nn.Conv2d(32, input_channels, 3, padding=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)



    def forward(self, x):
        skips = []

        x = self.enc0(x)
        skips.append(x)
        x = self.pool(x)

        x = self.enc1(x)
        skips.append(x)
        x = self.pool(x)

        x = self.enc2(x)
        skips.append(x)
        x = self.pool(x)

        x = self.enc3(x)
        skips.append(x)
        x = self.pool(x)

        x = self.enc4(x)
        skips.append(x)
        x = self.pool(x)

        x = self.enc5(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec5(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec4(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec3(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec2(x)

        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skips.pop()], dim=1)
        x = self.dec1(x)

        x = self.final_conv(x)
        # x = torch.sigmoid(x)
        return x
