import torch
import torch.nn as nn
from .MixedFeatureNet import MixedFeatureNet

class LinearBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
    def forward(self, x):
        return self.bn(self.conv(x))

class CoordAttHead(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        from .MixedFeatureNet import CoordAtt
        self.ca = CoordAtt(channels, channels)
    def forward(self, x):
        return self.ca(x)

class DDAMNet(nn.Module):
    def __init__(self, num_class=7, num_head=4):
        super().__init__()
        net = MixedFeatureNet()
        
        # We explicitly name these so they match the weights file keys perfectly
        self.conv1 = net.conv1
        self.conv2_dw = net.conv2_dw
        self.conv_23 = net.conv_23
        self.conv_3 = net.conv_3
        self.conv_34 = net.conv_34
        self.conv_4 = net.conv_4
        self.conv_45 = net.conv_45
        self.conv_5 = net.conv_5
        self.conv_6_sep = net.conv_6_sep

        self.num_head = num_head
        self.heads = nn.ModuleList([CoordAttHead(512) for _ in range(num_head)])
        
        # Shared linear/pool for the heads
        self.linear = LinearBlock(512, 512, groups=512, kernel=(7, 7))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 512 * 4 heads = 2048. Matches your [7, 2048] checkpoint error fix.
        self.fc = nn.Linear(512 * num_head, num_class)

    def forward(self, x):
        # Backbone
        x = self.conv1(x)
        x = self.conv2_dw(x)
        x = self.conv_23(x)
        x = self.conv_3(x)
        x = self.conv_34(x)
        x = self.conv_4(x)
        x = self.conv_45(x)
        x = self.conv_5(x)
        x_feat = self.conv_6_sep(x)

        # Head processing
        features = []
        for i in range(self.num_head):
            attn = self.heads[i](x_feat)
            y = x_feat * attn
            y = self.linear(y)
            y = self.pool(y)
            features.append(torch.flatten(y, 1))

        # Concatenate all heads: Result is size 2048
        combined = torch.cat(features, dim=1)
        return self.fc(combined)