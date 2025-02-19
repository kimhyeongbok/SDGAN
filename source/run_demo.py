from enum import Flag
import os
import sys
from unittest import result
import time

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__)))) # 현재 경로를 알아내어 상위폴더를 참조
# sys.path.append('C:/Users/rnd/PycharmProjects/SD')    # 절대 경로를 바로 하드 코딩하는 방식

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from os import listdir, mkdir
from os.path import isfile, join, isdir
import dlib
from PIL import Image
from numba import njit
import random
import torch
import torch.nn as nn
from torch.utils import data
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms
from EAGRNet.EAGR import EAGRNet
from EAGRNet.utils.transforms import get_affine_transform
from inplace_abn import InPlaceABN
from faceAttribute.resnet import *
import arch.arch_spade_feature_flex as arch_gen
Generator_model_path ='C:/Users/rnd/PycharmProjects/SD/models/Aspade_feature_flex_DCeleBA_Tcheck_ep01000G.pth'

dlib_path='/home/junjie/DeIDVideo/ciagan/source/'
EAGRNet_num_classes = 11
infer_time = []
def getFilePathList(file_dir):
    '''
    获取file_dir目录下，所有文本路径，包括子目录文件
    :param rootDir:
    :return:
    '''
    filePath_list = []
    for walk in os.walk(file_dir):
        part_filePath_list = [os.path.join(walk[0], file) for file in walk[2]]
        filePath_list.extend(part_filePath_list)
    return filePath_list

def get_files_list(file_dir, postfix=None):
    '''
    获得file_dir目录下，后缀名为postfix所有文件列表，包括子目录
    :param file_dir:
    :param postfix: ['*.jpg','*.png'],postfix=None表示全部文件
    :return:
    '''
    print('inner function : get file list')
    file_list = []
    filePath_list = getFilePathList(file_dir)
    print(filePath_list)
    if postfix is None:
        file_list = filePath_list
        print('inner function : get file list - None')
    else:
        postfix = [p.split('.')[-1] for p in postfix]
        print(filePath_list)
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name in postfix:
                file_list.append(file)
                print('inner function : get file list - else')
                print(file_list)

    file_list.sort()
    return file_list

def generate_clr(img_path,detector,predictor):
    res_w = 256
    res_h = 256
    gray = False
    img = cv2.imread(img_path,cv2.IMREAD_ANYCOLOR)

    if len(img.shape)==2:
        gray = True
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img0 = img.copy()

    img_dlib = np.array(img[:, :, :], dtype=np.uint8)
    dets = detector(img_dlib, 1)
    print("Number of faces detected: {}".format(len(dets)))
    clrs = []
    clr_cv2s = []
    cx = []
    cy = []
    hr = []
    wr = []
    pad = []
    img_pad = []
    visual_shape = []
    
    # for i in range(len(dets)-1):
    #     for j in range(len(dets)-i-1):
    #         if(dets[j].left()>=dets[j+1].left()):
    #             tmp = dets[j]
    #             dets[j] = dets[j+1]
    #             dets[j+1] = tmp
    for d in dets:
        landmarks = predictor(img_dlib, d)

        # centering
        c_x = int((landmarks.part(42).x + landmarks.part(39).x) / 2)
        c_y = int((landmarks.part(42).y + landmarks.part(39).y) / 2)
        w_r = int((landmarks.part(42).x - landmarks.part(39).x)*4)
        h_r = int((landmarks.part(42).x - landmarks.part(39).x)*5)
        w_r = int(h_r/res_h*res_w)
        # print(h_r)

        w, h = int(w_r * 2), int(h_r * 2)
        pd = int(w) # padding size
        img_p = np.zeros((img0.shape[0]+pd*2, img0.shape[1]+pd*2, 3), np.uint8) * 255
        img_p[:, :, 0] = np.pad(img0[:, :, 0], pd, 'edge')
        img_p[:, :, 1] = np.pad(img0[:, :, 1], pd, 'edge')
        img_p[:, :, 2] = np.pad(img0[:, :, 2], pd, 'edge')

        visual = img_p[c_y - h_r+pd:c_y + h_r+pd, c_x - w_r+pd:c_x + w_r+pd]
        clr = cv2.resize(visual, dsize=(res_w, res_h), interpolation=cv2.INTER_CUBIC)

        cx.append(c_x)
        cy.append(c_y)
        hr.append(h_r)
        wr.append(w_r)
        pad.append(pd)
        visual_shape.append(visual.shape[0:2])
        img_pad.append(img_p)
        clr_cv2s.append(clr)
        clrs.append(Image.fromarray(cv2.cvtColor(clr,cv2.COLOR_BGR2RGB)))

        # cv2.imwrite('C:/Users/rnd/PycharmProjects/SD/org.jpg',img)
        # cv2.imwrite('C:/Users/rnd/PycharmProjects/SD/img_p.jpg',img_p)
        # cv2.imwrite('C:/Users/rnd/PycharmProjects/SD/visual.jpg',visual)
        # cv2.imwrite('C:/Users/rnd/PycharmProjects/SD/clr.jpg',clr)

        # cv2.imwrite('/home/qiuyang/anonymous/ciagan_semantic/source/video/test_result/org.jpg', img)
        # cv2.imwrite('/home/qiuyang/anonymous/ciagan_semantic/source/video/test_result/img_p.jpg', img_p)
        # cv2.imwrite('/home/qiuyang/anonymous/ciagan_semantic/source/video/test_result/visual.jpg', visual)
        # cv2.imwrite('/home/qiuyang/anonymous/ciagan_semantic/source/video/test_result/clr.jpg', clr)

    return [cx,cy,hr,wr,pad],visual_shape,img_pad,img,clrs,clr_cv2s,gray

def generate_mask(semantic):
    img_msk = np.ones((semantic.shape[0], semantic.shape[1], 3), np.uint8) * 255
    
    for cls in range(1,10):
        index = np.where(semantic == cls)
        img_msk[index[0], index[1], :] =  0
    
    return Image.fromarray((img_msk).astype(np.uint8))
    
class EAGR_predictor():

    def __init__(self) -> None:
        self.crop_size = (256,256)
        self.aspect_ratio = self.crop_size[1] * 1.0 / self.crop_size[0]
        self.interp = torch.nn.Upsample(size=(128,128), mode='bilinear', align_corners=True)
        self.model = EAGRNet(EAGRNet_num_classes, InPlaceABN)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
        transforms.ToTensor(),
        self.normalize,
        ])

        restore_from = 'C:/Users/rnd/PycharmProjects/SD/source/EAGRNet/best.pth'
        # device = torch.device('gpu')
        # state_dict_old = torch.load(restore_from,map_location=device)
        # state_dict_old = torch.load(restore_from, map_location='cuda:0')
        state_dict_old = torch.load(restore_from)
        self.model.load_state_dict(state_dict_old, strict=False)
        # self.model.load_state_dict(state_dict_old)

        self.model.cuda()
        self.model.eval()



    def _xywh2cs(self,x, y, w, h):
        center = np.zeros((2), dtype=np.float32)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5
        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array([w * 1.0, h * 1.0], dtype=np.float32)

        return center, scale

    def _box2cs(self,box):
        x, y, w, h = box[:4]
        return self._xywh2cs(x, y, w, h)

    def getinput(self,im):
            
        h, w, _ = im.shape
            # Get center and scale
        center, s = self._box2cs([0, 0, w - 1, h - 1])
        r = 0
        trans = get_affine_transform(center, s, r, self.crop_size)
        input = cv2.warpAffine(
            im,
            trans,
            (int(self.crop_size[1]), int(self.crop_size[0])),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0))
            
        input = self.transform(input)
        return input

    def generate_semantic(self,img):
        img_input = self.getinput(img).unsqueeze(0)

        img_input = img_input.cuda()
        outputs,_ = self.model(img_input)

        parsing = self.interp(outputs).data.cpu().numpy()
        parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC

        image2 = img_input.cpu().numpy()
        image2 = image2.transpose(0, 2, 3, 1)  # NCHW NHWC
        return np.asarray(np.argmax(parsing, axis=3))[0],image2[0]

class new_model(nn.Module):
    def __init__(self,output_layer = None):
        super().__init__()
        # self.pretrained = resnet50(pretrained=True)
        self.pretrained = resnet50(pretrained=True, num_attributes=21)
        checkpoint = torch.load("C:/Users/rnd/PycharmProjects/SD/faceAttribute/checkpoints/model_best.pth.tar")
        self.pretrained.load_state_dict(checkpoint['state_dict'], strict=False)
        # self.pretrained.load_state_dict(checkpoint['state_dict'])
        self.output_layer = output_layer
        self.layers = list(self.pretrained._modules.keys())
        self.layer_count = 0
        for l in self.layers:
            if l != self.output_layer:
                self.layer_count += 1
            else:
                break
        for i in range(1,len(self.layers)-self.layer_count):
            self.dummy_var = self.pretrained._modules.pop(self.layers[-i])
        
        self.net = nn.Sequential(self.pretrained._modules)
        self.pretrained = None

    def forward(self,x):
        x = self.net(x)
        return x


class feature_predictor():
    def __init__(self):
        self.model = new_model(output_layer = 'avgpool')
        self.model.cuda()
        self.model.eval()
    def generate_feature(self,img):
        trans = transforms.Compose([
            transforms.Resize((128,128),interpolation=Image.NEAREST),
            transforms.ToTensor()
            ])
        input = trans(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))  )
        input = input.unsqueeze(0)
        output = self.model(input.cuda()).squeeze()
        output = output.detach().cpu().numpy()
        return output

# def generate_GAN_imput(clr,semantic,msk,feature):

def onehot_labelmap(img_mask):
#         print(img_mask.size())
    label_map = img_mask.unsqueeze(0)
#     print(label_map.size())
    bs, _, h, w = label_map.size()
    nc = 11
    input_label = torch.FloatTensor(bs, nc, h, w).zero_().cuda()
    input_semantics = input_label.scatter_(1, label_map, 1.0)
    return input_semantics

class GAN_Generator():
    def __init__(self):
        self.model = arch_gen.Generator()
        self.model.load_state_dict(torch.load(Generator_model_path, map_location=torch.device('cpu')))
        self.model.cuda()
        print('Generator is ready')


    def inference(self,im_faces, im_lndm, im_msk,im_feature,im_ind,num_classes = 1200):

        im_faces = im_faces.float()
        im_msk = im_msk.float()
        im_ind = im_ind.float()
        print('inference function')

        # num_classes = 1200
        output_id = (int(im_ind.cpu())+1)%num_classes
        print(output_id)
        hidden_feature_path = join('C:/Users/rnd/PycharmProjects/SD/CeleBA/clr/', str(output_id))
        # hidden_feature_path = join('/home/junjie/DeIDVideo/SemanticImageSynthesis/ciagan_semantic/CeleBA/clr/',str(output_id))
        # all_hidden_feature_names = glob.glob(hidden_feature_path+'/*feature_refined.npy')
        all_hidden_feature_names = glob.glob(hidden_feature_path+'/*feature.npy')
        hidden_feature = torch.from_numpy(np.load( all_hidden_feature_names[0] )).float().cuda()
        im_lndm = onehot_labelmap(im_lndm)

        with torch.no_grad():
            input_gen = torch.cat((im_lndm, im_faces * (1 - im_msk)), 1)
            bg = im_faces * (1 - im_msk)
            im_gen = self.model(im_lndm, bg, hidden_feature)
            im_gen = torch.clamp(im_gen*im_msk+im_faces * (1 - im_msk), 0, 1) #final image with BG
        
        img_out = transforms.ToPILImage()(im_gen[0].cpu()).convert("RGB")
        return img_out



def load_img( im, img_size = 128,flag_augment=True):
    crop_rnd = [random.random(), random.random(), random.random(), random.random()]
    if isinstance(img_size, tuple):
        img_shape = img_size
    else:
        img_shape = (img_size, img_size)

    transform_fnc=transforms.Compose([transforms.ToTensor()])

    im = im.resize([int(img_shape[0]*1.125)]*2, resample=Image.LANCZOS)
    im.save('C:/Users/rnd/PycharmProjects/SD/resized_im.jpg')
    w, h = im.size

    if flag_augment:
        offset_h = 0.
        center_h = h / 2 + offset_h * h
        center_w = w / 2
        min_sz, max_sz = w / 2, (w - center_w) * 1.5
        diff_sz, crop_sz = (max_sz - min_sz) / 2, min_sz / 2

        img_res = im.crop(
            (int(center_w - crop_sz - diff_sz * crop_rnd[0]), int(center_h - crop_sz - diff_sz * crop_rnd[1]),
                int(center_w + crop_sz + diff_sz * crop_rnd[2]), int(center_h + crop_sz + diff_sz * crop_rnd[3])))
    else:            
        offset_h = 0.
        center_h = h / 2 + offset_h * h
        center_w = w / 2
        min_sz, max_sz = w / 2, (w - center_w) * 1.5
        crop_sz = img_shape[0]/2
        img_res = im.crop(
            (int(center_w - crop_sz),
                int(center_h - crop_sz),
                int(center_w + crop_sz),
                int(center_h + crop_sz)))

    img_res = img_res.resize(img_shape, resample=Image.LANCZOS)
    img_res.save('C:/Users/rnd/PycharmProjects/SD/resized_mask.jpg')
#         print(img_res.shape)
    return transform_fnc(img_res)

def trans2input(clr,semantic,msk,feature,img_size = 128,flag_augment=True):
    im_clr, im_lndm, im_msk, im_ind, im_feature = [], [], [], [],[]

    clr.save('C:/Users/rnd/PycharmProjects/SD/clr.jpg')
    #clr
    im_clr.append(load_img(clr,flag_augment=flag_augment))
    print('trans2input')

    #semantic
    semantic_img = torch.from_numpy(semantic)
    im_lndm.append(semantic_img)

    #feature
    hidden_feature = torch.from_numpy(feature)
    im_feature.append(hidden_feature)

    #msk
    mask = ((1 - load_img(msk,flag_augment=flag_augment)) > 0.2)    
    im_msk.append(mask)

    return im_clr[0].unsqueeze(0).cuda(),im_lndm[0].unsqueeze(0).cuda(),im_msk[0].unsqueeze(0).cuda(),im_feature[0].unsqueeze(0).cuda()


def return2org(i,pos,visual_shape,img_pad,img_GAN,clr,img):
    img_GAN = cv2.cvtColor(np.asarray(img_GAN),cv2.COLOR_RGB2BGR)
    clr = cv2.resize(clr,(144,144),interpolation=cv2.INTER_CUBIC)
    clr[8:136,8:136] = img_GAN
    visual = cv2.resize(clr,visual_shape[i],interpolation=cv2.INTER_CUBIC)
    # img_p = img_pad[i]
    c_x = pos[0][i]
    c_y = pos[1][i]
    h_r = pos[2][i]
    w_r = pos[3][i]
    pd = pos[4][i]
    img_p = np.zeros((img.shape[0]+pd*2, img.shape[1]+pd*2, 3), np.uint8) * 255
    img_p[:, :, 0] = np.pad(img[:, :, 0], pd, 'edge')
    img_p[:, :, 1] = np.pad(img[:, :, 1], pd, 'edge')
    img_p[:, :, 2] = np.pad(img[:, :, 2], pd, 'edge')
    img_p[c_y - h_r+pd:c_y + h_r+pd, c_x - w_r+pd:c_x + w_r+pd] = visual
    img = img_p[pd:img_p.shape[0]-pd,pd:img_p.shape[1]-pd]
    cv2.imwrite('C:/Users/rnd/PycharmProjects/SD/source/video/test_result/return_img_p.jpg',img_p)
    cv2.imwrite('C:/Users/rnd/PycharmProjects/SD/source/video/test_result/return_img.jpg',img)
    return img



def run_inference(img_path,out_dir,dlib_detector,dlib_predictor,EAGR_model,feature_model,Generator,im_ind = 1,random_ind = False,multi=False):

    # start = time.time()
    pos,visual_shape,img_pad,img,clrs,clr_cv2s,gray=generate_clr(img_path,detector=dlib_detector,predictor=dlib_predictor)

    if multi:
        # pass
        org_name = os.path.basename(img_path).replace('.jpg','_org'+ '.jpg')
        org_path = join(out_dir,org_name)
        cv2.imwrite(org_path,img)
        print('multi')
        for i,(clr,clr_cv2) in enumerate(zip(clrs,clr_cv2s)):
            if i==1:
                break
            semantic,_ = EAGR_model.generate_semantic(clr_cv2)

            msk = generate_mask(semantic)

            feature = feature_model.generate_feature(clr_cv2)

            if random_ind:
                im_ind = random.randint(0, 1200)
            im_ind = torch.tensor(im_ind)
            im_clr, im_lndm, im_msk, im_feature = trans2input(clr,semantic,msk,feature,flag_augment=False)
            print(len(im_clr))

            img_out = Generator.inference(im_clr, im_lndm, im_msk, im_feature,im_ind)

            # end = time.time()
            # infer_time.append(end - start)

            result_name = os.path.basename(img_path).replace('.jpg', '_' + str(i) + '.jpg')
            result_path = join(out_dir,result_name)
            img_out.save(result_path)
            img = return2org(i,pos,visual_shape,img_pad,img_out,clr_cv2,img)

        # result_name = os.path.basename(img_path).replace('.jpg', '_our_epoch500.jpg')
        result_name = os.path.basename(img_path).replace('.jpg', '_our.jpg')
        # result_name = os.path.basename(img_path).replace('.jpg', '_trained_' + str(i) + '.jpg')
        result_path = join(out_dir,result_name)

        cv2.imwrite(result_path,img)

        
    else:
        last = '0'
        print('Not multi')

        for i,(clr,clr_cv2) in enumerate(zip(clrs,clr_cv2s)):
            if (i==1):
                break
            semantic,_ = EAGR_model.generate_semantic(clr_cv2)
            msk = generate_mask(semantic)
            feature = feature_model.generate_feature(clr_cv2)

            if random_ind:
                im_ind = random.randint(0,1200)
            # im_ind = int(img_path.split('/')[-2])
            im_ind = torch.tensor(im_ind)

            im_clr, im_lndm, im_msk, im_feature = trans2input(clr,semantic,msk,feature,flag_augment=False)
            img_out = Generator.inference(im_clr, im_lndm, im_msk, im_feature,im_ind)
            img_out.save('C:/Users/rnd/PycharmProjects/SD/img_gan.jpg')
            # end = time.time()
            # infer_time.append(end - start)
            # img_out.save('/home/qiuyang/anonymous/ciagan_semantic/source/video/test_result/img_gan.jpg')

            id = img_path.split('/')[-2]
            if not os.path.exists(join(out_dir,id)):
                os.mkdir(join(out_dir,id))
            result_name = os.path.basename(img_path).replace('.jpg', '_our3.jpg')
            # result_name = os.path.basename(img_path).replace('.jpg', '_trained_' + str(i) + '.jpg')
            result_path = join(out_dir,id,result_name)
            org_name = os.path.basename(img_path).replace('.jpg','_org'+ '.jpg')
            org_path = join(out_dir,id,org_name)
            if gray:
                img_out = img_out.convert('L')
            img_out.save(result_path)
            im_clr = transforms.ToPILImage()(im_clr[0].cpu()).convert("RGB")
            im_clr.save(org_path)


def vis_parsing_maps(im, parsing_anno, stride=1):
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                [255, 0, 85], [255, 0, 170],
                [0, 255, 0], [85, 255, 0], [170, 255, 0],
                [0, 255, 85], [0, 255, 170],
                [0, 0, 255], [85, 0, 255], [170, 0, 255],
                [0, 85, 255], [0, 170, 255],
                [255, 255, 0], [255, 255, 85], [255, 255, 170],
                [255, 0, 255], [255, 85, 255], [255, 170, 255],
                [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im

def test(img_path,out_dir,dlib_detector,dlib_predictor,EAGR_model,feature_model,Generator,im_ind = 1,random_ind = False):

    img,clrs,clr_cv2s=generate_clr(img_path,detector=dlib_detector,predictor=dlib_predictor)

    for i,(clr,clr_cv2) in enumerate(zip(clrs,clr_cv2s)):

        semantic,image = EAGR_model.generate_semantic(clr_cv2)
        im = vis_parsing_maps(image,semantic)

        result_name = os.path.basename(img_path).replace('.jpg', '_' + str(i) + '.jpg')
        result_path = join(out_dir,result_name)
        cv2.imwrite(im,result_path)

def prepare_traindata(EAGR_model):
    root_path = 'C:/Users/rnd/PycharmProjects/SD/dataset/Adience_train/clr'
    for dir in os.listdir(root_path):
        for name in os.listdir(os.path.join(root_path,dir)):
            im_path = os.path.join(root_path,dir,name)
            des_path = os.path.join(root_path,dir,name.replace('.jpg', '_semantic.npy'))
            clr = cv2.imread(im_path)
            semantic,_ = EAGR_model.generate_semantic(clr)
            np.save(des_path,semantic)


if __name__ == '__main__':
    # dir = '/home/qiuyang/anonymous/ciagan_semantic/dataset/FGNETT'
    dir = 'C:/Users/rnd/PycharmProjects/SD/dataset/FGNET/'
    # out_dir = 'C:/Users/rnd/PycharmProjects/SD/source/video/test_result/FGNET'
    out_dir = 'C:/Users/rnd/PycharmProjects/SD/result/FGNET/'

    dlib_detector = dlib.get_frontal_face_detector()
    dlib_predictor = dlib.shape_predictor("C:/Users/rnd/PycharmProjects/SD/source/shape_predictor_68_face_landmarks.dat")

    EAGR_model = EAGR_predictor()

    feature_model = feature_predictor()

    Generator = GAN_Generator()

    # run_inference(img_path,out_dir,dlib_detector,dlib_predictor,EAGR_model,feature_model,Generator,0,random_ind=False)

    if not os.path.isdir(out_dir):
         os.makedirs(out_dir)
    # file_list = get_files_list(dir,postfix=['*.jpg'])
    file_list = get_files_list(dir)

    im_ind = 0
    last_dir = '0'

    for img_path in file_list:
        if last_dir != img_path.split('/')[-2]:
            last_dir = img_path.split('/')[-2]
            im_ind += 1
            print(img_path)
        run_inference(img_path,out_dir,dlib_detector,dlib_predictor,EAGR_model,feature_model,Generator,im_ind,random_ind=False)

    # total_time = 0
    # for i in infer_time:
    #     total_time += i
    print("Done.")
    # print("FPS: %f"%(1.0/(total_time/len(infer_time))))
    # img_path = '/home/qiuyang/anonymous/ciagan_semantic/CeleBAT/orig/1/182757.jpg'
    # out_path = '/home/qiuyang/anonymous/ciagan_semantic/source/video/test_result'
    # # test(img_path,out_path,dlib_detector,dlib_predictor,EAGR_model,feature_model,Generator,random_ind=False)
    # prepare_traindata(EAGR_model)

    # img,clrs,clr_cv2s=generate_clr(img_path,detector=dlib_detector,predictor=dlib_predictor)













