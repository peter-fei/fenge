import os
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
LEARNING_RATE=1e-3
DEVICE='cuda' if torch.cuda.is_available() else 'gpu'
BATCH_SIZE=1
NUM_EPOCHS=20
IMG_HEIGHT=512
IMG_WIDTH=512
# IMG_HEIGHT=240
# IMG_WIDTH=192
# IMG_HEIGHT=208
# IMG_WIDTH=176
PIN_MEMORY=True
LOAD_MODEL=False
TRAIN=True
from net.deepv3 import DeepWV3Plus
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
TRAIN_IMG_DIR='../U-net-master/dataset/train/image'
TRAIN_MASK_DIR='../U-net-master/dataset/train/label'
TRAIN_EDGE_DIR='../U-net-master/dataset/train/edge2'
VAL_IMG_DIR='../U-net-master/dataset/test/image'
VAL_MASK_DIR='../U-net-master/dataset/test/label'
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
# VAL_IMG_DIR='data_Naso_crop2_a/crop1/test/image'
# VAL_MASK_DIR='data_Naso_crop1/test/label'
aaa=1
# 0.9477
# 0.5136

def train_fn(loader,model,optimizer:optim,loss_fn):
    loop=tqdm(loader)
    loss_t=0
    focal_loss=FocalLoss(logits=True)
    dice_loss=DiceLoss(logits=True)
    for index,(data,targets,edge) in enumerate(loop):

        data=data.type(torch.FloatTensor).to(DEVICE)
        targets=targets.unsqueeze(1).to(DEVICE)
        edge=edge.to(DEVICE)
        # edge=edge.to(DEVICE)
        # print(data.shape,targets.shape)
        # with torch.cuda.amp.autocast():
        # print(data.size(),targets.size(),edge.size())
        predictions,edge_outs=model(data,train=True)
        out=predictions
        # loss_body=IOU_loss(torch.sigmoid(body_out),targets)
        out=torch.sigmoid(out)
        loss_out=MSSSIM()(out,targets)+nn.BCELoss()(out,targets)+IOU_Loss()(out ,targets)
        loss_edge=0
        for k in range(len(edge_outs)-1):
            loss_edge += 0.5*(focal_loss(edge_outs[k], edge) +dice_loss(edge_outs[k],edge))
        loss_edge += 1.1*(focal_loss(edge_outs[-1], edge) +dice_loss(edge_outs[-1],edge))
        # loss_edge += 1.1 * (focal_loss(edge_outs, edge) + dice_loss(edge_outs, edge))
        loss=loss_out+loss_edge
        loss_t+=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()
        # scaler.step(optimizer)
        # scaler.update()
        loop.set_postfix(loss=loss.item(),loss_out=loss_out.item(),loss_edge=loss_edge.item())
    loss_t=loss_t/len(loop)
    print(f'loss is {loss_t}')

def main():
    train_trainsform=A.Compose([
        A.Resize(IMG_HEIGHT,IMG_WIDTH),
        # A.Rotate(limit=35,p=1.0),
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.1),
        A.Normalize(mean=(0,0,0),
                    std=(1.0,1.0,1.0),
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_trainsform = A.Compose([
        A.Resize(IMG_HEIGHT, IMG_WIDTH),
        A.Normalize(mean=(0, 0, 0),
                    std=(1.0, 1.0, 1.0),
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])
    edge_transform=transforms.Compose([
        transforms.Resize([IMG_HEIGHT,IMG_WIDTH]),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])
    if aaa==1:
        # model = M2(3, 1, (8, 16, 32, 64, 96)).cuda()
        # (4,16,32,48,64),(4,16,32,48,64)
        # model = M2(3, 1,(32,64,128,256,512)).cuda()
        model = M3(3, 1, (16, 32, 64, 72, 96)). cuda()
        # model.edge_net.load_state_dict(torch.load('checkpoints/save_edge_black.pth.tar')['state_dict'])
        # for p in model.edge_net.parameters():
        #     p.requires_grad=False
    # model=DenseNet(3,1,32,(6,12,36,16),32,0.5).to(DEVICE)
    # model = Unet(3, 1).to(DEVICE)
    # model = Dense_Unet(3, 1, 16, (2, 4, 8, 8, 8, 8, 4, 4), 4).cuda()
    elif aaa==2:
        # model=Dense_Unet_3plus(3,1,num_blocks=[4,8,16,24,32],growth_rate=4).cuda()
        # model=SKUnet_3plus(3,1,(32,64,72,96,128)).cuda()
        model=Unet(3,1,filters=[16,32,46,72,96]).cuda()
    loss_fn=MSSSIM()
    optimizer=optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),lr=LEARNING_RATE)

    train_loader,val_loader = get_loaders(TRAIN_IMG_DIR,
                                          TRAIN_MASK_DIR,
                                          VAL_IMG_DIR,
                                          VAL_MASK_DIR,
                                          TRAIN_EDGE_DIR,
                                          edge_transform,
                                          BATCH_SIZE,
                                          train_trainsform,
                                          val_trainsform,
                                          )
    max_dice = -1
    if LOAD_MODEL:
        if aaa==1:
            check=torch.load('save_pth/b2_final_3')
            # print(check['state_dict'])
            load_checkpoint(check, model)
            # max_dice=torch.load('save_pth/b2_final_2')['max_dice']
            max_dice = check['max_dice']
        elif aaa==2:
            load_checkpoint(torch.load('check6.pth.tar'),model)
    scaler=torch.cuda.amp.GradScaler()
    print(max_dice)
    for epoh in range(NUM_EPOCHS):
        if TRAIN:
            train_fn(train_loader,model,optimizer,loss_fn)

        # _=check_accury2(train_loader,model)
        dice=check_accury(val_loader,model)
        if dice>max_dice:
            max_dice=dice
            check_point = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'max_dice':max_dice
            }
            if aaa==1:
                print('----------------------savingmax---------------------')
            # save_checkpoint(check_point, filename='check3.pth.tar')
            #     save_checkpoint(check_point, filename='save_pth/b2_final_3')
            #     save_predictions_as_imgs7(val_loader,model,'save_imgs_bi3/')
                # save_checkpoint(check_point, filename='save_pth/gan_final_1')
                save_predictions_as_imgs7(val_loader,model,'save_imgs_gan1/')
                # save_predictions_as_imgs7(val_loader, model, 'save_imgs6_black/')
            elif aaa==2:
                save_checkpoint(check_point, filename='check6.pth.tar')
                save_predictions_as_imgs2(val_loader, model, 'save_imgs5/')

        check_point = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'max_dice': dice
        }
            # save_predictions_as_imgs7(val_loader, model, 'save_imgs_bi2/')
        # save_checkpoint(check_point, filename='save_pth/b2_final_3.pth.tar')
    print('max',max_dice)


if __name__=='__main__':
    main()

