import math
import torch
import torch.nn.functional as F
from torch import nn
from models import resnet
from torchvision import models
from base import BaseModel
from utils.helpers import initialize_weights, set_trainable
from itertools import chain

class _SpatialChannelTransformer(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, num_heads=4):
        super(_SpatialChannelTransformer, self).__init__()
        self.channel_attention = ChannelAttentionModule(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttentionModule(in_channels, num_heads)
        
    def forward(self, x):
        # Channel attention first
        x = self.channel_attention(x)
        # Then spatial attention
        x = self.spatial_attention(x)
        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        hidden_channels = max(in_channels // reduction_ratio, 1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        scale = self.sigmoid(out)
        return x * scale

class SpatialAttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super(SpatialAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, self.num_heads, self.head_dim, H * W)
        k = self.key(x).view(B, self.num_heads, self.head_dim, H * W)
        v = self.value(x).view(B, self.num_heads, self.head_dim, H * W)
        
        # Attention scores
        attn = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(self.head_dim)
        attn = self.softmax(attn)
        
        # Weighted sum
        out = torch.matmul(v, attn.transpose(-2, -1))
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        return x + out  # Residual connection

class SCTNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='resnet152', pretrained=True, 
                 use_aux=True, freeze_bn=False, freeze_backbone=False):
        super(SCTNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        model = getattr(resnet, backbone)(pretrained, norm_layer=norm_layer)
        m_out_sz = model.fc.in_features
        self.use_aux = use_aux 

        self.initial = nn.Sequential(*list(model.children())[:4])
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)
        
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        self.master_branch = nn.Sequential(
            _SpatialChannelTransformer(m_out_sz),
            nn.Conv2d(m_out_sz, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            _SpatialChannelTransformer(m_out_sz//2),
            nn.Conv2d(m_out_sz//2, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            norm_layer(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x_aux = self.layer3(x)
        x = self.layer4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')
        output = output[:, :, :input_size[0], :input_size[1]]

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            aux = aux[:, :, :input_size[0], :input_size[1]]
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                   self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

class SCTDenseNet(BaseModel):
    def __init__(self, num_classes, in_channels=3, backbone='densenet201', pretrained=True, 
                 use_aux=True, freeze_bn=False, **_):
        super(SCTDenseNet, self).__init__()
        self.use_aux = use_aux 
        model = getattr(models, backbone)(pretrained)
        m_out_sz = model.classifier.in_features
        aux_out_sz = model.features.transition3.conv.out_channels

        if not pretrained or in_channels != 3:
            block0 = [nn.Conv2d(in_channels, 64, 3, stride=2, bias=False), 
                      nn.BatchNorm2d(64), nn.ReLU(inplace=True)]
            block0.extend(
                [nn.Conv2d(64, 64, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True)] * 2
            )
            self.block0 = nn.Sequential(
                *block0,
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            initialize_weights(self.block0)
        else:
            self.block0 = nn.Sequential(*list(model.features.children())[:4])
        
        self.block1 = model.features.denseblock1
        self.block2 = model.features.denseblock2
        self.block3 = model.features.denseblock3
        self.block4 = model.features.denseblock4

        self.transition1 = model.features.transition1
        self.transition2 = nn.Sequential(*list(model.features.transition2.children())[:-1])
        self.transition3 = nn.Sequential(*list(model.features.transition3.children())[:-1])

        for n, m in self.block3.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (2,2), (2,2)
        for n, m in self.block4.named_modules():
            if 'conv2' in n:
                m.dilation, m.padding = (4,4), (4,4)

        self.master_branch = nn.Sequential(
            _SpatialChannelTransformer(m_out_sz),
            nn.Conv2d(m_out_sz, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        self.auxiliary_branch = nn.Sequential(
            _SpatialChannelTransformer(aux_out_sz),
            nn.Conv2d(aux_out_sz, m_out_sz//4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(m_out_sz//4),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(m_out_sz//4, num_classes, kernel_size=1)
        )

        initialize_weights(self.master_branch, self.auxiliary_branch)
        if freeze_bn: self.freeze_bn()

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        x = self.block0(x)
        x = self.block1(x)
        x = self.transition1(x)
        x = self.block2(x)
        x = self.transition2(x)
        x = self.block3(x)
        x_aux = self.transition3(x)
        x = self.block4(x_aux)

        output = self.master_branch(x)
        output = F.interpolate(output, size=input_size, mode='bilinear')

        if self.training and self.use_aux:
            aux = self.auxiliary_branch(x_aux)
            aux = F.interpolate(aux, size=input_size, mode='bilinear')
            return output, aux
        return output

    def get_backbone_params(self):
        return chain(self.block0.parameters(), self.block1.parameters(), self.block2.parameters(), 
                   self.block3.parameters(), self.transition1.parameters(), self.transition2.parameters(),
                   self.transition3.parameters())

    def get_decoder_params(self):
        return chain(self.master_branch.parameters(), self.auxiliary_branch.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
