import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            # nn.MaxPool2d(2),
            nn.AvgPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,  diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class PAM_Module(nn.Module):
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

#  Introducing 2 new modules - ASIB and TAAC for enhanced Novelty
class ASIB(nn.Module):
    """Memory-efficient Adaptive Scale Interaction Block"""
    def __init__(self, high_ch, low_ch, out_ch, reduction=2):
        super().__init__()
        self.pool = nn.AvgPool2d(reduction)

        self.query = nn.Conv2d(high_ch, out_ch, 1)
        self.key   = nn.Conv2d(low_ch, out_ch, 1)
        self.value = nn.Conv2d(low_ch, out_ch, 1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, high_feat, low_feat):
        high_ds = self.pool(high_feat)
        low_ds  = self.pool(low_feat)

        B, C, H, W = high_ds.shape

        q = self.query(high_ds).view(B, -1, H*W).permute(0, 2, 1)
        k = self.key(low_ds).view(B, -1, H*W)
        v = self.value(low_ds).view(B, -1, H*W)

        attn = self.softmax(torch.bmm(q, k))
        out  = torch.bmm(v, attn.permute(0, 2, 1))
        out  = out.view(B, -1, H, W)

        out = F.interpolate(out, size=high_feat.shape[2:], mode='bilinear', align_corners=False)
        return high_feat + self.gamma * out

class TAAC(nn.Module):
    """
    Task-Aware Attention Calibration
    Channel recalibration conditioned on decoder semantic context
    """
    def __init__(self, x_ch, task_ch, reduction=4):
        super().__init__()

        self.task_proj = nn.Linear(task_ch, x_ch)

        self.fc = nn.Sequential(
            nn.Linear(x_ch * 2, x_ch // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(x_ch // reduction, x_ch),
            nn.Sigmoid()
        )

    def forward(self, x, task_feat):
        b, c, _, _ = x.size()

        # Global pooling
        x_gap = F.adaptive_avg_pool2d(x, 1).view(b, c)
        t_gap = F.adaptive_avg_pool2d(task_feat, 1).view(b, -1)

        # Project task features to x channel space
        t_gap = self.task_proj(t_gap)

        # Channel attention
        attn = self.fc(torch.cat([x_gap, t_gap], dim=1)).view(b, c, 1, 1)

        return x * attn
