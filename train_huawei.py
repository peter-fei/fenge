import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import albumentations as A
from tqdm import tqdm
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
import torch.optim as optim
from torchvision import transforms
from util.utils import *

from net.mod2 import *




import sys
sys.path.append('D:\zhiwensuo\pytorch\loss')
# print(sys.path)
from loss.msssimLoss import *
from loss.iouLoss import IOU_Loss
from loss.diceloss import DiceLoss
from loss.edge_loss import *

# 0.9444
# 0.5314
LEARNING_RATE=1e-3
DEVICE='cuda' if torch.cuda.is_available() else 'gpu'
BATCH_SIZE=1
NUM_EPOCHS=10
IMG_HEIGHT=512
IMG_WIDTH=512
PIN_MEMORY=True
TRAIN=True
from net.deepv3 import DeepWV3Plus
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import argparse
aaa=1
# 0.9477
# 0.5136

# GSUNet0 0.5990 (16,32,64,72,96)
# GSUNet 0.9391 (16,48,64,72,96)
if TRAIN==True:
    NUM_EPOCHS=1


def train_fn(loader,model,optimizer,loss_fn=None,mutiedges=False):
    loop=tqdm(loader)
    loss_ssim=SSIM_Loss()
    loss_bce=nn.BCELoss()
    focal_loss = FocalLoss(logits=True)
    dice_loss = DiceLoss(logits=True)
    losses=0
    for index,(data,targets,edge) in enumerate(loop):

        data=data.type(torch.FloatTensor).to(DEVICE)
        targets=targets.unsqueeze(1).to(DEVICE)
        edge=edge.to(DEVICE)
        predictions=model(data,train=True)
        # out,edge_out=predictions
        # out,body,edge_outs=predictions
        out, edge_outs = predictions
        out=torch.sigmoid(out)
        loss_out=loss_ssim(out,targets)+loss_bce(out,targets)

        # loss_edge=focal_loss(edge_outs,edge)+dice_loss(edge_    outs,edge)


        # loss_body = focal_loss(body, targets) + dice_loss(body, targets)
        loss_edge=0
        if mutiedges:
            for k in range(len(edge_outs)-1):
                loss_edge += 0.5 * (focal_loss(edge_outs[k], edge) +dice_loss(edge_outs[k],edge))
            loss_edge += 1.1 * (focal_loss(edge_outs[-1], edge) +dice_loss(edge_outs[-1],edge))
        else:
            loss_edge += 1.1 * (focal_loss(edge_outs, edge) + dice_loss(edge_outs, edge))
        loss=loss_out+loss_edge
        losses+=loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # loop.set_postfix(loss=loss.item(), loss_out=loss_out.item(), loss_edge=loss_edge.item(),loss_body=loss_body.item())
        # loop.set_postfix(loss=loss.item(),loss_out=loss_out.item(),loss_edge=loss_edge.item())
    print(f'loss is {losses/len(loader)}')

def main(args):
    train_trainsform=A.Compose([
        A.Resize(args.height,args.width),
        A.Normalize(mean=(0,0,0),
                    std=(1.0,1.0,1.0),
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])
    val_trainsform = A.Compose([
        A.Resize(args.height, args.width),
        A.Normalize(mean=(0, 0, 0),
                    std=(1.0, 1.0, 1.0),
                    max_pixel_value=255.0),
        ToTensorV2(),
    ])
    edge_transform=transforms.Compose([
        transforms.Resize([args.height, args.width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])
    if args.mod=='M1':
        model=M1(3,1,(args.f1,args.f2,args.f3,args.f4,args.f5)).cuda()
        mutiedges = True
    elif args.mod=='M2':
        model=M2(3,1,(args.f1,args.f2,args.f3,args.f4,args.f5)).cuda()
        mutiedges=True
    elif args.mod=='M3':
        model=M3(3,1,(args.f1,args.f2,args.f3,args.f4,args.f5)).cuda()
        mutiedges = False
    elif args.mod=='M4':
        model=M4(3,1,(args.f1,args.f2,args.f3,args.f4,args.f5)).cuda()
        mutiedges = True
    elif args.mod=='M5':
        model=M5(3,1,(args.f1,args.f2,args.f3,args.f4,args.f5)).cuda()
        mutiedges = True
    elif args.mod=='M6':
        model=M6(3,1,(args.f1,args.f2,args.f3,args.f4,args.f5)).cuda()
        mutiedges = False
    loss_fn=MSSSIM()
    optimizer=optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),lr=args.lr)

    train_loader = get_loaders(args.TRAIN_IMG_DIR,
                                          args.TRAIN_MASK_DIR,
                                          args.TRAIN_EDGE_DIR,
                                          edge_transform,
                                          args.batch_size,
                                          train_trainsform,
                                          )
    # val_loader=get_loaders(args.VAL_IMG_DIR,
    #                        args.VAL_MASK_DIR,
    #                        batch_size=args.batch_size,
    #                        transform=val_trainsform
    #                        )
    max_dice = -1
    if args.load:
        checkpoint=torch.load(args.checkpoint)
        load_checkpoint(checkpoint, model)
        max_dice=checkpoint['max_dice']
    print(max_dice)
    for epoh in range(args.epohs):
        if args.train:
            train_fn(train_loader,model,optimizer,loss_fn,mutiedges)

        _=check_accury(train_loader,model,hasedge=True)
        dice = 0
        c = 0
        for f in os.listdir(args.VAL_IMG_DIR):
            val_loader2 = get_loaders(os.path.join(args.VAL_IMG_DIR,f), os.path.join(args.VAL_MASK_DIR,f),batch_size=args.batch_size,transform=val_trainsform)
            c+=1
            dice+=check_accury2(val_loader2,model)
        dice/=c
        print('train dice ',_)
        print('test dice',dice)
        # dice=check_accury_noloop(val_loader,model)
        if dice>max_dice:
            max_dice=dice
            check_point = {
                'state_dict': model.state_dict(),
                'max_dice':max_dice
            }
            if aaa==1:
            # save_checkpoint(check_point, filename='check3.pth.tar')
                save_checkpoint(check_point, filename=os.path.join(args.savepath,args.checkpoint))
                # save_predictions_as_imgs7(val_loader,model,args.savepath)
                # save_predictions_as_imgs7(val_loader, model, args.savepath)
        check_point = {
            'state_dict': model.state_dict(),
            'max_dice': max_dice
        }
        save_checkpoint(check_point,'./save1.pth.tar')
    print('max',max_dice)


if __name__=='__main__':
    parse=argparse.ArgumentParser()
    # parse.add_argument('--TRAIN_IMG_DIR',default='data_Naso2/train/image')
    # parse.add_argument('--TRAIN_MASK_DIR',default='data_Naso2/train/label')
    # parse.add_argument('--VAL_IMG_DIR',default='data_Naso2/test/image')
    # parse.add_argument('--VAL_MASK_DIR',default='data_Naso2/test/label')
    # parse.add_argument('--TRAIN_EDGE_DIR',default='data_Naso2/train/edge2')

    parse.add_argument('--mod',default='M4')
    parse.add_argument('--TRAIN_IMG_DIR', default='dataset/train/image')
    parse.add_argument('--TRAIN_MASK_DIR', default='dataset/train/label')
    parse.add_argument('--VAL_IMG_DIR', default='dataset/test/image')
    parse.add_argument('--VAL_MASK_DIR', default='dataset/test/label')
    parse.add_argument('--TRAIN_EDGE_DIR', default='dataset/train/edge2')
    parse.add_argument('--batch_size',type=int,default=1)
    parse.add_argument('--height', type=int, default=512)
    parse.add_argument('--width', type=int, default=512)
    parse.add_argument('--lr',default=1e-3,type=float)
    parse.add_argument('--load',default=0)
    parse.add_argument('--train',default=True)
    parse.add_argument('--save',default='save_predictions_as_imgs8')
    parse.add_argument('--savepath',default='save_imgs')
    parse.add_argument('--checkpoint',default='save.pth.tar')
    parse.add_argument('--epohs',default=10,type=int)
    # parse.add_argument('--mutiedges',default=False)

    parse.add_argument('--f1', default=4, type=int)
    parse.add_argument('--f2', default=6, type=int)
    parse.add_argument('--f3', default=8, type=int)
    parse.add_argument('--f4', default=12, type=int)
    parse.add_argument('--f5', default=24, type=int)
    args=parse.parse_args()
    # args.load=True
    args.res2=True
    print(args.mod,args.f1,args.f2,args.f3,args.f4,args.f5)
    main(args)

