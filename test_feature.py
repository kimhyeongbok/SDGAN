#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: maximov
"""

import torch
import numpy as np
import util_func
import util_data
from torchvision import transforms, utils
import os
from os.path import join
from os import listdir
import arch.arch_spade_feature_flex as arch_gen
import glob
import argparse
import time

def onehot_labelmap(img_mask, device_comp):
#         print(img_mask.size())
    label_map = img_mask.unsqueeze(0)
#     print(label_map.size())
    bs, _, h, w = label_map.size()
    nc = 11
    input_label = torch.FloatTensor(bs, nc, h, w).zero_().to(device_comp)
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return input_semantics
    
    
def inference(generator, out_dir, data_loader, device_comp, num_classes = 1200):
    total_imgs = 0
    infer_time = []
    for batch in data_loader:
        torch.cuda.synchronize()
        start = time.time()
        # prepare data
        im_faces, im_lndm, im_msk, im_ind, _ = [item[0].to(device_comp) for item in batch]
        print(im_lndm.size)
        im_faces = im_faces.float()
        im_msk = im_msk.float()
        im_ind = im_ind.float()
        output_id = (int(im_ind[0].cpu())+1)%num_classes #chose next id
        
        hidden_feature_path = join('/home/junjie/DeIDVideo/SemanticImageSynthesis/ciagan_semantic/CeleBA/clr/', str(output_id))
        # hidden_feature_path = join('/home/qiuyang/anonymous/ciagan_semantic/CeleBA/clr/', str(output_id))
        # all_hidden_feature_names = glob.glob(hidden_feature_path+'/*feature.npy')
        all_hidden_feature_names = glob.glob(hidden_feature_path+'/*feature_refined.npy')
        hidden_feature = torch.from_numpy(np.load( all_hidden_feature_names[0] )).float().to(device_comp)
        im_lndm = onehot_labelmap(im_lndm, device_comp)
        

        # inference
        with torch.no_grad():
            input_gen = torch.cat((im_lndm, im_faces * (1 - im_msk)), 1)
            bg = im_faces * (1 - im_msk)
            im_gen = generator(im_lndm, bg, hidden_feature)
            im_gen = torch.clamp(im_gen*im_msk+im_faces * (1 - im_msk), 0, 1) #final image with BG
            

        torch.cuda.synchronize()
        end = time.time()
        infer_time.append(end - start) 
        # output image
        # img_out = transforms.ToPILImage()(im_gen[0].cpu()).convert("RGB")
        # img_out.save(join(out_dir, str(total_imgs).zfill(6) + '.jpg'))
        # img_out = transforms.ToPILImage()(im_faces[0].cpu()).convert("RGB")
        # img_out.save(join(out_dir, str(total_imgs).zfill(6) + '_org.jpg'))
       
        dir = join(out_dir,str(int(im_ind)))
        if not os.path.exists(dir):
            os.mkdir(dir)
        img_out = transforms.ToPILImage()(im_gen[0].cpu()).convert("RGB")
        img_out.save(join(out_dir[:-4], str(total_imgs).zfill(6) + '.jpg'))
        img_out.save(join(dir, str(total_imgs).zfill(6) + '.jpg'))
        img_out = transforms.ToPILImage()(im_faces[0].cpu()).convert("RGB")
        img_out.save(join(out_dir[:-4], str(total_imgs).zfill(6) + '_org.jpg'))


        total_imgs+=1
    total_time = 0
    for i in infer_time:
        total_time += i
    print("Done.")
    print("FPS: %f"%(1.0/(total_time/len(infer_time))))



def run_inference(data_path='../dataset/celeba/', num_folders = -1, model_path = '../modelG', output_path = '../output'):
    ##### PREPARING DATA
    if num_folders==-1:
        num_folders = len(listdir(join(data_path,'lndm')))

    dataset_test = util_data.ImageDataset(root_dir=data_path, label_num=num_folders, transform_fnc=transforms.Compose([transforms.ToTensor()]), flag_sample=1, flag_augment = False)
    data_loader = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)



    ##### PREPARING MODELS
    device_comp = util_func.set_comp_device(True)
    model = arch_gen.Generator()
    model.load_state_dict(torch.load(model_path + '.pth', map_location=torch.device('cpu')))
    model.to(device_comp)
    print('Model is ready')
    
    inference(model, output_path, data_loader, device_comp=device_comp)



# Instantiate the parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='path to input data directory', default='../CeleBAT/')
parser.add_argument('--ids', type=int, help='how many folder/ids to process', default=-1)
parser.add_argument('--model', type=str, help='path to a pre-trained model and its name', default='/home/junjie/DeIDVideo/SemanticImageSynthesis/ciagan_semantic/models/ciagan_Aspade_feature_flex_DCeleBA_Tcheckhyperparameters_att__unitloss/Aspade_feature_flex_DCeleBA_Tcheck_ep01000G')
parser.add_argument('--out', type=str, help='path to output data directory', default='../CeleBAT/feature_out_ind')

args = parser.parse_args()


run_inference(data_path=args.data, num_folders=args.ids,model_path=args.model, output_path=args.out)



 