import torch
import torch.nn as nn
from torch.nn import Module
from PIL import Image
import numpy as np
from torchvision.transforms import ToTensor

class DiceLoss(Module):
    def __init__(self,logits=False):
        super(DiceLoss, self).__init__()
        self.logits=logits
    def forward(self,pred,label):
        if self.logits:
            pred=torch.sigmoid(pred)
        n=pred.size(0)
        pred=pred.reshape(n,-1)
        label=label.reshape(n,-1)
        dice_score=(2*torch.sum(pred*label,dim=1)+1e-8)/(torch.sum(pred,dim=1)+torch.sum(label,dim=1)+1e-8)
        dice_score=dice_score.sum()/n
        return 1-dice_score

class ROILoss(nn.Module):
    def __init__(self,beta=3,logits=False):
        super(ROILoss, self).__init__()
        self.beta=beta
        self.logits=logits
    def forward(self,preds,targets):
        if self.logits:
            preds=torch.sigmoid(preds)
        bias_dice=self.get_bias_dice(preds,targets)
        return 1-bias_dice
    def get_bias_dice(self,preds,targets):
        e=1e-8
        n=preds.size(0)
        preds=preds.reshape(n,-1)
        targets=targets.reshape(n,-1)
        tp=torch.sum(preds*targets)
        fp=torch.sum(preds*(1-targets))
        fn=torch.sum(targets*(1-preds))
        dice=2*(tp+e)/(fp+2*tp+self.beta*fn+e)/n
        return dice
if __name__=='__main__':
    img=np.array(Image.open('D:/zhiwensuo/pytorch/U-net-master/dataset/train/label/000_mask.png').convert('L'))
    transpose=ToTensor()
    img=transpose(img).unsqueeze(0)
    print(img.shape,torch.max(img),torch.min(img))
    loss_fn=ROILoss()
    print(loss_fn(img,img))
