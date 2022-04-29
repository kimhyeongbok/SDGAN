import torch 
from arch.arch_spade_feature_flex import *


if __name__ == '__main__':
    gnet = Generator()
    
    semantic_input = torch.ones(( 2, 11, 128, 128  ))
    bg_input = torch.ones(( 2,3,128,128 ))
    onehot_input = torch.zeros((2, 2048))

    output = gnet(semantic_input, bg_input, onehot_input)