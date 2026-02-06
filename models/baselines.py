import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=[64,128,256,512]):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        # Encoder
        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Decoder
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(f*2, f))

        self.final_conv = nn.Conv2d(features[0], num_classes, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        decoder_feature_maps = {}
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skip_connections[idx//2]

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)

        # Save the decoder feature map after this stage
        decoder_feature_maps[f'decoder_stage_{idx//2 + 1}'] = x

        out = self.final_conv(x)
        decoder_feature_maps['output'] = out
        
        return out, decoder_feature_maps

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, padding=0),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, features=[64,128,256,512]):
        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.attentions = nn.ModuleList()

        ch = in_channels
        for f in features:
            self.downs.append(DoubleConv(ch, f))
            ch = f

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.attentions.append(AttentionGate(F_g=f, F_l=f, F_int=f//2))
            self.ups.append(DoubleConv(f*2, f))

        self.final_conv = nn.Conv2d(features[0], num_classes, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        skips = []

        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        att_idx = 0
        decoder_feature_maps = {}
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip = skips[idx//2]

            skip = self.attentions[att_idx](x, skip)
            att_idx += 1

            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)
            x = self.ups[idx+1](x)

        # Save the decoder feature map after this stage
        decoder_feature_maps[f'decoder_stage_{idx//2 + 1}'] = x
        
        out = self.final_conv(x)
        decoder_feature_maps['output'] = out
        
        return out, decoder_feature_maps

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1, img_size=256, patch_size=16,
                 embed_dim=512, num_heads=8, depth=4):

        super().__init__()

        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = DoubleConv(64, 128)
        self.encoder3 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.patch_embed = nn.Conv2d(256, embed_dim, kernel_size=patch_size, stride=patch_size)

        num_patches = (img_size // 8 // patch_size) ** 2

        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, embed_dim*4) for _ in range(depth)]
        )

        self.unpatch = nn.ConvTranspose2d(embed_dim, 256, patch_size, patch_size)

        self.decoder1 = DoubleConv(512, 256)
        self.decoder2 = DoubleConv(384, 128)
        self.decoder3 = DoubleConv(192, 64)

        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        e1 = self.encoder1(x)
        e2 = self.encoder2(self.pool(e1))
        e3 = self.encoder3(self.pool(e2))

        x = self.pool(e3)

        x = self.patch_embed(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = x + self.pos_embed[:, :x.size(1)]

        x = self.transformer(x)

        x = x.transpose(1,2).reshape(B, C, H, W)
        x = self.unpatch(x)

        decoder_feature_maps = {}
        
        x = F.interpolate(x, size=e3.shape[2:])
        x = self.decoder1(torch.cat([x, e3], dim=1))
        decoder_feature_maps['decoder_stage_1'] = x

        x = F.interpolate(x, size=e2.shape[2:])
        x = self.decoder2(torch.cat([x, e2], dim=1))
        decoder_feature_maps['decoder_stage_2'] = x

        x = F.interpolate(x, size=e1.shape[2:])
        x = self.decoder3(torch.cat([x, e1], dim=1))
        decoder_feature_maps['decoder_stage_3'] = x

        out = self.final(x)
        decoder_feature_maps['output'] = out
        
        return out, decoder_feature_maps
