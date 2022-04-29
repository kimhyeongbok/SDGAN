import numpy as np
import cv2
import os
import json
from collections import OrderedDict
import argparse
from PIL import Image as PILImage
from utils.transforms import transform_parsing
from tqdm import tqdm

LABELS = ['background', 'skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """

    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def get_confusion_matrix(gt_label, pred_label, num_classes):
    """
    Calcute the confusion matrix by given label and pred
    :param gt_label: the ground truth label
    :param pred_label: the pred label
    :param num_classes: the nunber of class
    :return: the confusion matrix
    """
    index = (gt_label * num_classes + pred_label).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for i_label in range(num_classes):
        for i_pred_label in range(num_classes):
            cur_index = i_label * num_classes + i_pred_label
            if cur_index < len(label_count):
                confusion_matrix[i_label, i_pred_label] = label_count[cur_index]

    return confusion_matrix

def fast_histogram(a, b, na, nb):
    '''
    fast histogram calculation
    ---
    * a, b: non negative label ids, a.shape == b.shape, a in [0, ... na-1], b in [0, ..., nb-1]
    '''
    assert a.shape == b.shape
    assert np.all((a >= 0) & (a < na) & (b >= 0) & (b < nb))
    # k = (a >= 0) & (a < na) & (b >= 0) & (b < nb)
    hist = np.bincount(
        nb * a.reshape([-1]).astype(int) + b.reshape([-1]).astype(int),
        minlength=na * nb).reshape(na, nb)
    assert np.sum(hist) == a.size
    return hist


def _read_names(file_name):
    label_names = []
    for name in open(file_name, 'r'):
        name = name.strip()
        if len(name) > 0:
            label_names.append(name)
    return label_names


def _merge(*list_pairs):
    a = []
    b = []
    for al, bl in list_pairs:
        a += al
        b += bl
    return a, b


def compute_mean_ioU(preds, scales, centers, num_classes, datadir, input_size=[473, 473], dataset='val', reverse=False):
    file_list_name = os.path.join(datadir, dataset + '/files.txt')
    val_id = [line.strip() for line in open(file_list_name).readlines()]

    confusion_matrix = np.zeros((num_classes, num_classes))

    label_names_file = os.path.join(datadir, 'label_names.txt')
    gt_label_names = pred_label_names = _read_names(label_names_file)

    assert gt_label_names[0] == pred_label_names[0] == 'bg'

    hists = []
    for i, im_name in enumerate(val_id):
        gt_path = os.path.join(datadir, dataset , 'labels', im_name + '.png')
        
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        h, w = gt.shape
        pred_out = preds[i]
        s = scales[i]
        c = centers[i]
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        if reverse:
            proj = np.load(os.path.join(datadir, 'project', im_name + '.npy'))
            pred = cv2.warpAffine(pred, proj, (gt.shape[1], gt.shape[0]), borderValue=0, flags = cv2.INTER_NEAREST)

        gt = np.asarray(gt, dtype=np.int32)
        pred = np.asarray(pred, dtype=np.int32)

        ignore_index = gt != 255

        gt = gt[ignore_index]
        pred = pred[ignore_index]

        hist = fast_histogram(gt, pred,
                              len(gt_label_names), len(pred_label_names))
        hists.append(hist)

        confusion_matrix += get_confusion_matrix(gt, pred, num_classes)

    hist_sum = np.sum(np.stack(hists, axis=0), axis=0)

    eval_names = dict()
    for label_name in gt_label_names:
        gt_ind = gt_label_names.index(label_name)
        pred_ind = pred_label_names.index(label_name)
        eval_names[label_name] = ([gt_ind], [pred_ind])
    if 'le' in eval_names and 're' in eval_names:
        eval_names['eyes'] = _merge(eval_names['le'], eval_names['re'])
    if 'lb' in eval_names and 'rb' in eval_names:
        eval_names['brows'] = _merge(eval_names['lb'], eval_names['rb'])
    if 'ulip' in eval_names and 'imouth' in eval_names and 'llip' in eval_names:
        eval_names['mouth'] = _merge(
            eval_names['ulip'], eval_names['imouth'], eval_names['llip'])
    if 'eyes' in eval_names and 'brows' in eval_names and 'nose' in eval_names and 'mouth' in eval_names:
        eval_names['overall'] = _merge(
            eval_names['eyes'], eval_names['brows'], eval_names['nose'], eval_names['mouth'])
    print(eval_names)


    pos = confusion_matrix.sum(1)
    res = confusion_matrix.sum(0)
    tp = np.diag(confusion_matrix)

    pixel_accuracy = (tp.sum() / pos.sum()) * 100
    mean_accuracy = ((tp / np.maximum(1.0, pos)).mean()) * 100
    IoU_array = (tp / np.maximum(1.0, pos + res - tp))
    IoU_array = IoU_array * 100
    mean_IoU = IoU_array.mean()
    print('Pixel accuracy: %f \n' % pixel_accuracy)
    print('Mean accuracy: %f \n' % mean_accuracy)
    print('Mean IU: %f \n' % mean_IoU)
    mIoU_value = []
    f1_value = []

    for i, (label, iou) in enumerate(zip(LABELS, IoU_array)):
        mIoU_value.append((label, iou))

    for eval_name, (gt_inds, pred_inds) in eval_names.items():
        A = hist_sum[gt_inds, :].sum()
        B = hist_sum[:, pred_inds].sum()
        intersected = hist_sum[gt_inds, :][:, pred_inds].sum()
        f1 = 2 * intersected / (A + B)
        print(f'f1_{eval_name}={f1}')
        f1_value.append((eval_name, f1))

    mIoU_value.append(('Pixel accuracy', pixel_accuracy))
    mIoU_value.append(('Mean accuracy', mean_accuracy))
    mIoU_value.append(('Mean IU', mean_IoU))
    mIoU_value = OrderedDict(mIoU_value)
    f1_value = OrderedDict(f1_value)
    return mIoU_value, f1_value

def write_results(preds, scales, centers, datadir, dataset, result_dir, input_size=[473, 473]):
    palette = get_palette(20)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    json_file = os.path.join(datadir, 'annotations', dataset + '.json')
    with open(json_file) as data_file:
        data_list = json.load(data_file)
        data_list = data_list['root']
    for item, pred_out, s, c in zip(data_list, preds, scales, centers):
        im_name = item['im_name']
        w = item['img_width']
        h = item['img_height']
        pred = transform_parsing(pred_out, c, s, w, h, input_size)
        #pred = pred_out
        save_path = os.path.join(result_dir, im_name[:-4]+'.png')

        output_im = PILImage.fromarray(np.asarray(pred, dtype=np.uint8))
        output_im.putpalette(palette)
        output_im.save(save_path)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV NetworkEv")
    parser.add_argument("--pred-path", type=str, default='',
                        help="Path to predicted segmentation.")
    parser.add_argument("--gt-path", type=str, default='',
                        help="Path to the groundtruth dir.")

    return parser.parse_args()