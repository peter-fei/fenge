import torch
import torchvision
from torch.utils.data import DataLoader
from tqdm import tqdm
import PIL.Image as Image
from torch.utils.data import Dataset
import os
import numpy as np

class Data_load(Dataset):
    def __init__(self,img_dir,mask_dir,transform=None):
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.imgs=os.listdir(img_dir)
        self.masks=os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path=os.path.join(self.img_dir,self.imgs[index])
        mask_path=os.path.join(self.mask_dir,self.masks[index])
        image=np.array(Image.open(img_path).convert('RGB'))
        mask=np.array(Image.open(mask_path).convert('L'),dtype=np.float32)
        mask/=255
        if self.transform is not None:
            augmentation=self.transform(image=image,mask=mask)
            image=augmentation['image']
            mask=augmentation['mask']
        return image,mask


def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    print('====================>saving')
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print('========>loading')
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(train_dir,train_mask_dir,val_dir,val_mask_dir,batch_size,train_transform,val_transform,pin_nemory=True):
    train_ds=Data_load(train_dir,train_mask_dir,train_transform)
    val_ds=Data_load(val_dir,val_mask_dir,val_transform)

    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,pin_memory=pin_nemory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_nemory)
    return train_loader,val_loader

def check_accury(loader,model,device='cuda'):
    num_correct=0
    num_pixels=0
    dice_score=0
    model.eval()
    loop=tqdm(loader)
    with torch.no_grad():
        for x,y in loop:
            x=x.to(device)
            y=y.to(device)
            preds=torch.sigmoid(model(x))
            preds=(preds>0.5).float()

            num_correct+=(preds==y).sum()
            num_pixels+=torch.numel(preds)
            dice_score+=(2*(preds*y).sum())/((preds+y).sum()+1e-11)

    print(f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}')
    print('Dice_score:',dice_score/len(loader))
    model.train()
    return dice_score/len(loader)

def save_predictions_as_imgs(loader,model,folder='save_imgs/'):
    model.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.to('cuda')
        with torch.no_grad():
            preds=torch.sigmoid(model(x))
            preds=(preds>0.5).float()
        torchvision.utils.save_image(
            preds,f'{folder}/{idx}_pred.png'
        )
        torchvision.utils.save_image(y,f'{folder}/{idx}.png')
    model.train()






