from genericpath import isdir
import os
from pathlib import Path
import shutil
src_dir = '/home/qiuyang/anonymous/ciagan_semantic/CeleBAT/clr'
des_dir = '/home/qiuyang/anonymous/ciagan_semantic/dataset/test'
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
    file_list = []
    filePath_list = getFilePathList(file_dir)
    if postfix is None:
        file_list = filePath_list
    else:
        postfix = [p.split('.')[-1] for p in postfix]
        for file in filePath_list:
            basename = os.path.basename(file)  # 获得路径下的文件名
            postfix_name = basename.split('.')[-1]
            if postfix_name in postfix:
                file_list.append(file)
    file_list.sort()
    return file_list

file_list = get_files_list(src_dir,postfix=['*.jpg'])
for src_path in file_list:
    dir  = Path(os.path.join(des_dir,src_path.split('/')[-2]))
    if not dir.is_dir():
        os.mkdir(dir)
    des_path = os.path.join(des_dir,src_path.split('/')[-2],src_path.split('/')[-1])
    shutil.copyfile(src_path, des_path)