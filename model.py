# models.py
import torch
from torch import nn

class BatchNormRelu(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        
        self.batch_norm = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(True)
    
    def forward(self, input):
        x = self.batch_norm(input)
        x = self.relu(x)
        return x
    
class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.batnorm_relu1 = BatchNormRelu(in_channels)
        self.conv_in1 = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  stride=stride,
                                  bias=False)
        self.batnorm_relu2 = BatchNormRelu(out_channels)
        self.conv_in2 = nn.Conv2d(out_channels,
                                  out_channels,
                                  kernel_size=3,
                                  padding=1,
                                  stride=1)
        
        self.identity_map = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=1,
                                     stride=stride)
    
    def forward(self, inputs):
        x = self.batnorm_relu1(inputs)
        x = self.conv_in1(x)
        x = self.batnorm_relu2(x)
        x = self.conv_in2(x)
        s = self.identity_map(inputs)
        
        skip = x + s
        
        return skip
    
class decoder_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.res_block = residual_block(in_channels + out_channels, out_channels)
        
    def forward(self, inputs, skip):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.res_block(x)
        return x

#Res Unet
class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # encoder 1
        self.conv_in1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.batnorm_relu = BatchNormRelu(64)
        self.conv_in2 = nn.Conv2d(64, 64, kernel_size=3, padding = 1)
        self.identity_map = nn.Conv2d(in_channels, 64, kernel_size=1)
        
        # encoder 2
        self.res_block2 = residual_block(64, 128, stride=2)
        # encoder 3
        self.res_block3 = residual_block(128, 256, stride=2)
        
        # bridge
        self.res_blockb = residual_block(256, 512, stride=2)
        
        # decoder 3
        self.dec_block3 = decoder_block(512, 256)
        # decoder 2
        self.dec_block2 = decoder_block(256, 128)
        # decoder 1
        self.dec_block1 = decoder_block(128, 64)
        
        # output
        self.output = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, inputs):
        # encoder 1
        x = self.conv_in1(inputs)
        x = self.batnorm_relu(x)
        x = self.conv_in2(x)
        s = self.identity_map(inputs)
        
        e1 = x + s
        # encoder 2
        e2 = self.res_block2(e1)
        # encoder 3
        e3 = self.res_block3(e2)
        
        # bridge
        bridge = self.res_blockb(e3)
        
        # decoder 3
        d3 = self.dec_block3(bridge, e3)
        # decoder 2
        d2 = self.dec_block2(d3, e2)
        # decoder 1
        d1 = self.dec_block1(d2, e1)
        
        # output
        output = self.output(d1)
        return output
    
class PatchDiscriminator(nn.Module):
    """不包含输出层激活函数
    激活函数内嵌在损失函数中"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.batchNorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=False)
        self.batchNorm4 = nn.BatchNorm2d(512)
        # 最后一层的Channel降为1，以输出一张N*N的单层真假性集合
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)

        self.leakyRelu = nn.LeakyReLU(0.2, True)

    def forward(self, x, label):
        x = torch.cat([x, label], 1)
        x = self.leakyRelu(self.conv1(x))
        x = self.leakyRelu(self.batchNorm2(self.conv2(x)))
        x = self.leakyRelu(self.batchNorm3(self.conv3(x)))
        x = self.leakyRelu(self.batchNorm4(self.conv4(x)))
        # sigmoid内嵌在损失函数中
        return self.conv5(x)
