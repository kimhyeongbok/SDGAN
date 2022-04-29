import numpy as np
from PIL import Image
im = Image.open('/home/qiuyang/anonymous/ciagan_semantic/CeleBAT/clr/3/006383.jpg')
a = np.asarray(im)
label2color_dict = {
                    0:[0,0,0],
                    1:[200,0,0],
                    2:[250,0,150],
                    3:[200,150,150], 
                    4:[250,150,150],
                    5:[0,200,0],
                    6:[150,250,0],
                    7:[150,200,150],
                    8:[200,0,200],
                    9:[150,0,250],
                    10:[150,150,250],
                    11:[250,200,0],
                    12:[200,200,0],
                    13:[0,0,200],
                    14:[0,150,200],
                    15:[0,200,250]
                    }
# data_path = '/home/qiuyang/anonymous/ciagan_semantic/CeleBAT/clr/3/006383_mask.npy'
# data =  np.load(data_path)
# print(data.shape)
# img = np.zeros((256,256,3), dtype = 'uint8')
# for i in range(256):
#     for j in range(256):
#         if data[i][j] != 10 and data[i][j] != 0:
#             img[i][j] = [0,0,0]
#         else:
#             img[i][j] = a[i][j]
# image = Image.fromarray(img)
# image.save('./test_mask.jpg')      
# 
data_path = '/home/qiuyang/anonymous/ciagan_semantic/CeleBA/clr/1/182659_semantic.npy'   
data =  np.load(data_path)
img = np.zeros((128,128,3), dtype = 'uint8')
for i in range(128):
    for j in range(128):
        img[i][j] = label2color_dict[data[i][j]]
print(img)
image = Image.fromarray(img)
image.save('./182659_semantic.jpg')