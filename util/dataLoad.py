import PIL.Image as Image
import torch
from torch.utils.data import Dataset
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision import transforms

class Data_load(Dataset):
    def __init__(self,img_dir,mask_dir,edge_dir=None,transform=None,edge_trans=None):
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.imgs=os.listdir(img_dir)
        self.masks=os.listdir(self.mask_dir)
        self.has_edge=False
        self.edge_trans=edge_trans
        if edge_dir is not None:
            self.edge_dir=edge_dir
            self.edges=os.listdir(edge_dir)
            self.has_edge=True

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
        if self.has_edge:
            edge_path=os.path.join(self.edge_dir,self.edges[index])
            edge = Image.open(edge_path).convert('L')
            if self.edge_trans:
                edge=self.edge_trans(edge)
                if torch.max(edge)>1:
                    edge/=255
            return image,mask,edge
        return image,mask

class Data_load_Gray(Dataset):
    def __init__(self,img_dir,mask_dir,edge_dir=None,transform=None,edge_trans=None):
        self.img_dir=img_dir
        self.mask_dir=mask_dir
        self.transform=transform
        self.imgs=os.listdir(img_dir)
        self.masks=os.listdir(self.mask_dir)
        self.has_edge=False
        self.edge_trans=edge_trans
        if edge_dir is not None:
            self.edge_dir=edge_dir
            self.edges=os.listdir(edge_dir)
            self.has_edge=True

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_path=os.path.join(self.img_dir,self.imgs[index])
        mask_path=os.path.join(self.mask_dir,self.masks[index])
        image=np.array(Image.open(img_path).convert('L'),dtype=np.float32)
        mask=np.array(Image.open(mask_path).convert('L'),dtype=np.float32)
        image/=255
        mask/=255
        if self.transform is not None:
            augmentation=self.transform(image=image,mask=mask)
            image=augmentation['image']
            mask=augmentation['mask']
        if self.has_edge:
            edge_path=os.path.join(self.edge_dir,self.edges[index])
            edge = Image.open(edge_path).convert('L')
            if self.edge_trans:
                edge=self.edge_trans(edge)
                if torch.max(edge)>1:
                    edge/=255
            return image,mask,edge
        return image,mask

if __name__=='__main__':
    # a=Data_load('../../data_Naso/image','../../data_Naso/label')
    edge_trans=transforms.Compose([
        transforms.Resize((512,512)),
        transforms.ToTensor()
    ])
    trans=A.Compose([
        A.Resize(512,512),
        ToTensorV2()
    ])
    a = Data_load('../../U-net-master/dataset/train/image', '../../U-net-master/dataset/train/label','../../U-net-master/dataset/train/edge',transform=trans,edge_trans=edge_trans)
    x,y,z=a[0]
    print(x.shape,y.shape,z.shape,torch.max(z))
    # print(x[x>0])
    # print((x==0).sum(),(x>0).sum(),(x==255).sum(),512*512*3)
    # print(np.max(x),np.min(x),np.max(y),np.min(y))