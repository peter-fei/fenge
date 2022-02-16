import os

from util.dataLoad import Data_load

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torchvision import transforms
from util.utils import load_checkpoint, save_checkpoint, get_loaders, check_accury, save_predictions_as_imgs, \
    save_predictions_as_imgs2, save_predictions_as_imgs7
from torch.utils.data import DataLoader

# from Dense_Unet import Dense_Unet
# from DenseNet import DenseNet
#from dense_unet_plus import Dense_Unet_3plus
# from  dense_unet_plus2 import Dense_Unet_3plus
from net.att_unet_3plus import Att_Unet_3plus
from net.SKunet_3plus import SKUnet_3plus
from net.mod2 import *
from net.double_net import Doubel_Net
import PIL.Image as Image
import matplotlib.pyplot as plt

import sys
sys.path.append('D:\zhiwensuo\pytorch\loss')
# print(sys.path)
from loss.msssimLoss import MSSSIM
from loss.iouLoss import IOU_Loss
from loss.edge_loss import FocalLoss
from loss.diceloss import DiceLoss
# 0.9444
# 0.5314
LEARNING_RATE=1e-4
DEVICE='cuda' if torch.cuda.is_available() else 'gpu'
BATCH_SIZE=16
NUM_EPOCHS=50
# IMG_HEIGHT=512
# IMG_WIDTH=512
# IMG_HEIGHT=240
# IMG_WIDTH=192
IMG_HEIGHT=128
IMG_WIDTH=112
# IMG_HEIGHT=208
# IMG_WIDTH=176
PIN_MEMORY=True
LOAD_MODEL=True
TRAIN=False
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# TRAIN_IMG_DIR='../U-net-master/dataset/train/image'
# TRAIN_MASK_DIR='../U-net-master/dataset/train/label'
# TRAIN_EDGE_DIR='../U-net-master/dataset/train/edge2'
# VAL_IMG_DIR='../U-net-master/dataset/test/image'
# VAL_MASK_DIR='../U-net-master/dataset/test/label'
# TRAIN_IMG_DIR='save_imgs_roi'
# TRAIN_MASK_DIR='save_imgs_roi_label'
# TRAIN_EDGE_DIR='save_imgs_roi_edge'
# VAL_IMG_DIR='save_imgs_roi'
# VAL_MASK_DIR='save_imgs_roi_label'
# TRAIN_IMG_DIR='preprocess-npz-data/train/image'
# TRAIN_MASK_DIR='preprocess-npz-data/train/label'
# TRAIN_EDGE_DIR='preprocess-npz-data/train/edge2'
# VAL_IMG_DIR='preprocess-npz-data/test/image'
# VAL_MASK_DIR='preprocess-npz-data/test/label'
# TRAIN_IMG_DIR='data_Naso_crop2_a/crop1/train/image'
# TRAIN_MASK_DIR='data_Naso_crop1/train/label'
# TRAIN_EDGE_DIR='data_Naso_crop1/train/edge'
# VAL_IMG_DIR='data_Naso_crop2_a/crop1/train/image'
# VAL_MASK_DIR='data_Naso_crop1/train/label'
# VAL_IMG_DIR='data_Naso_crop2_a/crop1/test/image'
# VAL_MASK_DIR='data_Naso_crop1/test/label'


VAL_IMG_DIR='data_Naso_crop2_a/crop3/test/image'
VAL_MASK_DIR='data_Naso_total_crop/test/label'


def main():
    val_trainsform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0, 0, 0),
                    std=(1.0, 1.0, 1.0),
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])

    model = M4(3, 1, (32, 64, 128, 256, 512)).cuda()
    val_ds = Data_load(VAL_IMG_DIR, VAL_MASK_DIR, transform=val_trainsform)
    test_datas=DataLoader(val_ds,BATCH_SIZE,False,pin_memory=True)

    max_dice = -1
    if LOAD_MODEL:
        check = torch.load('save_pth/bi_final_m4.pth.tar')
        # print(check['state_dict'])
        load_checkpoint(check, model)
        # max_dice=torch.load('save_pth/b2_final_2')['max_dice']
        max_dice = check['max_dice']
        # max_dice =-1
    scaler = torch.cuda.amp.GradScaler()
    print(max_dice)
    for epoh in range(NUM_EPOCHS):
        dice = check_accury(test_datas, model)
        print('test Dice_score:', dice)
        save_predictions_as_imgs7(test_datas, model, 'save_imgs_bi_4/')

    print('max', max_dice)

if __name__=='__main__':
    main()