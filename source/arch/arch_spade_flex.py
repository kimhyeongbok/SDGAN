#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 14:54:12 2018

@author: maximov
"""


import torch
import torch.nn as nn
import torch.utils.data
from torch.nn import functional as F

from arch.base_network import BaseNetwork
from arch.normalization import get_nonspade_norm_layer
from arch.architecture import ResnetBlock as ResnetBlock
from arch.architecture import SPADEResnetBlock as SPADEResnetBlock
# from base_network import BaseNetwork
# from normalization import get_nonspade_norm_layer
# from architecture import ResnetBlock as ResnetBlock
# from architecture import SPADEResnetBlock as SPADEResnetBlock


class Args():
    num_upsampling_layers = 'normal'
    ngf = 64
    
    norm_G = 'spectralspadesyncbatch3x3'
    semantic_nc = 11
    ndf = 64
    output_nc = 3
    label_nc = 11
    no_instance = True
      
    


# main architecture. use concatenation
class Generator(nn.Module):

    def __init__(self, input_nc=11, num_classes=1200, encode_one_hot = True, img_size=128, **kwargs):
        super(Generator, self).__init__()

        self.in_dim = input_nc
        self.encode_one_hot = encode_one_hot
        self.img_size = img_size
        opt =Args()

        # align the back ground with semantic
        self.align_bg_conv = nn.Conv2d(3, input_nc, 3, padding=1)
        input_ch = input_nc
        # follow SPADE ResNet
        if img_size==128:
            self.conv0 = SPADEResnetBlock(input_ch, 32, opt)
            input_ch = 32


        self.conv1 = SPADEResnetBlock(input_ch, 64, opt)
        self.conv2 = SPADEResnetBlock(64, 128, opt)
        self.conv3 = SPADEResnetBlock(128,256, opt)
        self.conv4 = SPADEResnetBlock(256, 256, opt)

        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)

        # embed onehot with image
        self.embed = nn.Sequential(
            ConvLayer(512, 256, kernel_size=3, stride=1),
            nn.InstanceNorm2d(256, affine=True),
        )

        self.up = nn.Upsample(scale_factor=2)
        self.deconv4 = SPADEResnetBlock(256, 256, opt)
        self.deconv3 = SPADEResnetBlock(256, 128, opt)
        self.deconv2 = SPADEResnetBlock(128, 64, opt)
        self.deconv1 = SPADEResnetBlock(64, 32, opt)
         
        if img_size == 128:
            self.deconv0 = SPADEResnetBlock(32, 16, opt)

        self.conv_end = nn.Sequential(nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1),)

        self.flag_onehot = encode_one_hot
        if encode_one_hot:
            self.encode_one_hot = nn.Sequential(
                nn.Linear(num_classes, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 256), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(256, 512), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(512, 1024), nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(1024, 2048), nn.LeakyReLU(0.2, inplace=True),
                #nn.LeakyReLU(0.2, inplace=True),
            )
            self.encode_noise = nn.Sequential(
                ConvLayer(32, 64, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(64, affine=True),
                ConvLayer(64, 128, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(128, affine=True),
                ConvLayer(128, 256, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(256, affine=True),
            )
        else:
            self.encode_one_hot = None


    def convblock(self, in_ch,out_ch, krn_sz = 3):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=krn_sz, stride=1, padding=int(krn_sz/2)),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return block


    def forward(self, seg, bg, onehot=None, high_res=0):
        
        # step 1: encode bg to align semantic
        # step 2: encode semantic?
        # step 3: extract latent with bg and semantic
        # step 4: upsampling

        bg = self.align_bg_conv(bg)


        # Encode
        if self.img_size==128:
            
            out =  self.conv0(bg, seg)

        # print(out.size(), seg.size())
        out =  self.conv1(out, seg)  # [B, 64, 32, 32]
        out = F.avg_pool2d(out, 2)
        # seg = F.avg_pool2d(seg, 2)

        # print(out.size(), seg.size())
        out =  self.conv2(out, seg)  # [B, 128, 16, 16]
        out = F.avg_pool2d(out, 2)
        # seg = F.avg_pool2d(seg, 2)

        # print(out.size(), seg.size())
        out =  self.conv3(out, seg)  # [B, 256, 8, 8]
        out = F.avg_pool2d(out, 2)
        # seg = F.avg_pool2d(seg, 2)

        # print(out.size(), seg.size())
        out =  self.conv4(out, seg)  # [B, 256, 4, 4]
        out = F.avg_pool2d(out, 2)
        # seg = F.avg_pool2d(seg, 2)

        # print(out.size(), seg.size())

        # Embedding
        if onehot is not None and self.flag_onehot:
            noise = self.encode_one_hot(onehot)
            noise = noise.view(-1, 32, 8, 8)
            noise = self.encode_noise(noise)
            # print(noise.size(), out.size())
            out = torch.cat((out, noise), 1)
            out = self.embed(out)

        # Residual layers
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)

        # Decode
        out = self.up(out)
        out = self.deconv4(out,seg)  # [B, 256, 8, 8]
        out = self.up(out)
        out = self.deconv3(out,seg)  # [B, 128, 16, 16]
        out = self.up(out)
        out = self.deconv2(out,seg)  # [B, 64, 32, 32]
        out = self.deconv1(out,seg)  # [B, 32, 64, 64]
        # print(out.size())
        if self.img_size==128:
            out = self.deconv0(out, seg)
            out = self.up(out)
            # print(out.size())
#         print(self.img_size, out.size())
        out = self.conv_end(out) # [B, 3, 64, 64]
        #out = torch.sigmoid(out)
        # print(out.size())
        return out


class Discriminator(nn.Module):
    def __init__(self, input_nc=3, num_classes=1200, img_size=64, **kwargs):
        super(Discriminator, self).__init__()

        self.img_size = img_size

        self.conv1 = ResidualBlockDown(input_nc, 64)
        self.conv2 = ResidualBlockDown(64, 128)
        self.conv3 = ResidualBlockDown(128, 256)
        self.conv4 = ResidualBlockDown(256, 512)
        if img_size==128:
            self.conv5 = ResidualBlockDown(512, 512)

        self.dense0 = nn.Linear(8192, 1024)
        self.dense1 = nn.Linear(1024, 1)

    def forward(self, x, high_res=0):
        out = x  # [B, 6, 64, 64]
        # Encode
        out_0 = (self.conv1(out))  # [B, 64, 32, 32]
        out_1 = (self.conv2(out_0))  # [B, 128, 16, 16]
        out_3 = (self.conv3(out_1))  # [B, 256, 8, 8]
        out = (self.conv4(out_3))  # [B, 512, 4, 4]
        if self.img_size==128:
            out = (self.conv5(out))  # [B, 512, 4, 4]

        out = out.view(out.size(0), -1)
        out = F.leaky_relu(self.dense0(out), 0.2, inplace=True)
        out = F.leaky_relu(self.dense1(out), 0.2, inplace=True)
        return out


# region Residual Blocks
class ResidualBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None):
        super(ResidualBlockDown, self).__init__()

        # Right Side
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride, padding)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride, padding)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = F.relu(x)
        out = self.conv_r1(out)
        out = F.relu(out)
        out = self.conv_r2(out)
        out = F.avg_pool2d(out, 2)

        # Left Side
        residual = self.conv_l(residual)
        residual = F.avg_pool2d(residual, 2)

        # Merge
        out = residual + out
        return out


class ResidualBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, upsample=2):
        super(ResidualBlockUp, self).__init__()

        # General
        self.upsample = nn.Upsample(scale_factor=upsample, mode='nearest')

        # Right Side
        self.norm_r1 = nn.InstanceNorm2d(in_channels, affine=True)
        self.conv_r1 = ConvLayer(in_channels, out_channels, kernel_size, stride)

        self.norm_r2 = nn.InstanceNorm2d(out_channels, affine=True)
        self.conv_r2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

        # Left Side
        self.conv_l = ConvLayer(in_channels, out_channels, 1, 1)

    def forward(self, x):
        residual = x

        # Right Side
        out = self.norm_r1(x)
        out = F.relu(out)
        out = self.upsample(out)
        out = self.conv_r1(out)
        out = self.norm_r2(out)
        out = F.relu(out)
        out = self.conv_r2(out)

        # Left Side
        residual = self.upsample(residual)
        residual = self.conv_l(residual)

        # Merge
        out = residual + out
        return out


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.in1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.in2(out)

        out = out + residual
        return out


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=None):
        super(ConvLayer, self).__init__()
        if padding is None:
            padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding)
        self.conv2d = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size, stride))

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out
# endregion


# if __name__ == '__main__':
#     gnet = Generator()
    
#     semantic_input = torch.ones(( 2, 11, 128, 128  ))
#     bg_input = torch.ones(( 2,3,128,128 ))
#     onehot_input = torch.zeros((2, 1200))

#     output = gnet(semantic_input, bg_input, onehot_input)
