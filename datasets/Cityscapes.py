from __future__ import division
import os.path
from .listdataset import  ListDataset

import numpy as np
import flow_transforms
from pathlib import Path

try:
    import cv2
except ImportError as e:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("default", category=ImportWarning)
        warnings.warn("failed to load openCV, which is needed"
                      "for KITTI which uses 16bit PNG images", ImportWarning)

'''
Data load for bsds500 dataset:
author:Fengting Yang 
Mar.1st 2019

usage:
1. manually change the name of train.txt and val.txt in the make_dataset(dir) func.    
2. ensure the val_dataset using the same size as the args. in the main code when performing centerCrop 
   default value is 320*320, it is fixed to be 16*n in our project
'''

# mode: train or val or test
def make_dataset_path(dir, mode):
    # we train and val seperately to tune the hyper-param and use all the data for the final training
    # train_list_path = os.path.join(dir, 'train.txt') # use train_Val.txt for final report
    # val_list_path = os.path.join(dir, 'val.txt')

    # try:
    #     with open(train_list_path, 'r') as tf:
    #         train_list = tf.readlines()
    #
    #     with open (val_list_path, 'r') as vf:
    #         val_list = vf.readlines()
    #
    # except IOError:
    #     print ('Error No avaliable list ')
    #     return

    image_root = os.path.join(dir, 'leftImg8bit')
    image_path_list = sorted(Path(os.path.join(image_root, mode)).glob('**/*.png'))
    label_root = os.path.join(dir, 'gtFine')
    label_path_list = sorted(Path(os.path.join(label_root, mode)).glob('**/*_labelIds.png'))

    return image_path_list, label_path_list



def BSD_loader(path_imgs, path_label):
    # cv2.imread is faster than io.imread usually
    img = cv2.imread(path_imgs)[:, :, ::-1].astype(np.float32)
    gtseg = cv2.imread(path_label)[:,:,:1]

    return img, gtseg


# root = './datasets'
def Cityscapes(root, transform=None, target_transform=None, val_transform=None,
              co_transform=None, split=None):


    mode = 'train'
    train_image_path_list, train_label_path_list = make_dataset_path(root, mode)
    mode = 'val'
    val_image_path_list, val_label_path_list = make_dataset_path(root, mode)

    print(train_image_path_list)


    # if val_transform ==None:
    #     val_transform = transform
    #
    # train_dataset = ListDataset(root, 'bsd500', train_list, transform,
    #                             target_transform, co_transform,
    #                             loader=BSD_loader, datatype = 'train')
    #
    # val_dataset = ListDataset(root, 'bsd500', val_list, val_transform,
    #                            target_transform, flow_transforms.CenterCrop((320,320)),
    #                            loader=BSD_loader, datatype = 'val')
    #
    # return train_dataset, val_dataset


if __name__ == '__main__':
    Cityscapes('../datasets/Cityscapes')
