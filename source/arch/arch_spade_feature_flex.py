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
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        try:
            avg_out = torch.mean(x, dim=1, keepdim=True)
            max_out, _ = torch.max(x, dim=1, keepdim=True)
            scale = torch.cat([avg_out, max_out], dim=1)
            scale = self.conv(scale)
            out = x * self.sigmoid(scale)
        except Exception as e:
            print(e)
            out = x

        return out

class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1 ,L=32):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        d = max(int(features/r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=stride, padding=1+i, groups=G),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

class SKUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(SKUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features/2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.InstanceNorm2d(mid_features),
            SKConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.InstanceNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.InstanceNorm2d(out_features)
        )
        if in_features == out_features: # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else: # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.InstanceNorm2d(out_features)
            )
    
    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)
# 2021-12-26 13:17 remove residual layer from decoder, order to remove the high level fusion from the fake feature Attention: need to be set Args.use_spatial_att=False
#                  model path: SemanticImageSynthesis/ciagan_semantic/models/ciagan_Aspade_feature_flex_DCeleBA_Tcheckhyperparameters
# 2021-12-27 10:29 pm add SpatialAttention to improve intra-class semantic consistency : Attention: need to be set Args.use_spatial_att=True
#                  model path: SemanticImageSynthesis/ciagan_semantic/models/ciagan_Aspade_feature_flex_DCeleBA_Tcheckhyperparameters_att

# 2022-01-07 9:51 am add residual layer for decoder, use SpatialAttention, use loss/loss.detach()
#                 model path: SemanticImageSynthesis/ciagan_semantic/models/ciagan_Aspade_feature_flex_DCeleBA_Tcheckhyperparameters_att_unitloss

# 2022-01-07 9:14 pm use SKnet for encode noise, because the sknet can adaptive select the kernel size and the noise actually is the feature map to guide the generator. I 
# want the encode_noise sub_module can select adaptive the importance features
#                 model path: SemanticImageSynthesis/ciagan_semantic/models/ciagan_Aspade_feature_flex_DCeleBA_Tcheckhyperparameters_att_unitloss_sknoise
class Args():
    num_upsampling_layers = 'normal'
    ngf = 64
    
    norm_G = 'spectralspadesyncbatch3x3'
    semantic_nc = 11
    ndf = 64
    output_nc = 3
    label_nc = 11
    no_instance = True
    use_spatial_att=True
    encode_noise_sk=True
      
    


# main architecture. use concatenation
class Generator(nn.Module):

    def __init__(self, input_nc=11,   img_size=128,  **kwargs):
        super(Generator, self).__init__()

        self.in_dim = input_nc
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
        
        self.noise_encode_type = opt.encode_noise_sk
        if self.noise_encode_type  == True:
            self.encode_noise = nn.Sequential(
                ConvLayer(32, 64, kernel_size=3, stride=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.InstanceNorm2d(64, affine=True),
                SKUnit(64, 128, 32, 2, 8, 2, stride=1),
                nn.LeakyReLU(0.2, inplace=True), 
                SKUnit(128, 256, 32, 2, 8, 2, stride=1), 
                nn.LeakyReLU(0.2, inplace=True), 
                # SKUnit(256, 256, 32, 2, 8, 2, stride=1),
                # nn.LeakyReLU(0.2, inplace=True),
            )
        else:

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
        self.use_spatial_att = opt.use_spatial_att
        if self.use_spatial_att:
            self.spatialAtt = SpatialAttention()
         


    def convblock(self, in_ch,out_ch, krn_sz = 3):
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=krn_sz, stride=1, padding=int(krn_sz/2)),
            #nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
        return block


    def forward(self, seg, bg, hidden_feature=None, high_res=0):
        
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
        if hidden_feature is not None:
            noise = hidden_feature.view(-1, 32, 8, 8)
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
        if self.use_spatial_att:
            x = self.spatialAtt(out)
            out = F.leaky_relu(x, 2e-1)

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
