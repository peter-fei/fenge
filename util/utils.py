import torch
import torchvision
from util.dataLoad import Data_load,Data_load_Gray
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
# from util.save_edge import save_edge

def save_checkpoint(state,filename='my_checkpoint.pth.tar'):
    path,name=os.path.split(filename)
    if not os.path.exists(path):
        os.mkdir(path)
        print('creating path {}'.format(path))
    print('====================>saving')
    torch.save(state,filename)

def load_checkpoint(checkpoint,model):
    print('========>loading')
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(train_dir,train_mask_dir,val_dir,val_mask_dir,train_edge_dir=None,train_edge_transform=None,batch_size=1,train_transform=None,val_transform=None,pin_nemory=True,load_type='RGB'):
    if load_type=='RGB':
        train_ds =Data_load(train_dir, train_mask_dir, edge_dir=train_edge_dir, transform=train_transform,
                                  edge_trans=train_edge_transform)
        val_ds = Data_load(val_dir, val_mask_dir, transform=val_transform)
    elif load_type=='L':
        train_ds=Data_load_Gray(train_dir,train_mask_dir,edge_dir=train_edge_dir,transform=train_transform,edge_trans=train_edge_transform)
        val_ds=Data_load_Gray(val_dir,val_mask_dir,transform=val_transform)

    train_loader=DataLoader(train_ds,batch_size=batch_size,shuffle=True,pin_memory=pin_nemory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, pin_memory=pin_nemory)
    return train_loader,val_loader

def check_accury(loader,model,device='cuda',res=False,res2=False,softmax=False):
    num_correct=0
    num_pixels=0
    dice_score=0
    model.eval()
    loop=tqdm(loader)
    count=0
    with torch.no_grad():
        for x,y in loop:
            x=x.type(torch.FloatTensor).to(device)
            y=y.to(device).unsqueeze(1)
            preds=model(x)
            if softmax:
                preds=torch.softmax(preds,dim=1)
                preds=torch.argmax(preds,dim=1)
            if res:
                preds=model.res
            elif res2:
                preds=preds[0]

            preds=torch.sigmoid(preds)
            preds=(preds>0.5).float()

            num_correct+=(preds==y).sum()
            num_pixels+=torch.numel(preds)
            if (preds+y).sum()==0:
                continue
            else:
                dice_score+=(2*(preds*y).sum())/((preds+y).sum())
                count+=1

    print(f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}')
    print('Dice_score:',dice_score/count)
    model.train()
    return (dice_score+1e-6)/(count+1e-6)

def check_accury_noloop(loader,model,device='cuda',res=False,res2=False,softmax=False):
    num_correct=0
    num_pixels=0
    dice_score=0
    model.eval()
    count=0
    with torch.no_grad():
        for x,y in loader:
            x=x.type(torch.FloatTensor).to(device)
            y=y.to(device).unsqueeze(1)
            preds=model(x)
            if softmax:
                preds=torch.softmax(preds,dim=1)
                preds=torch.argmax(preds,dim=1)
            if res:
                preds=model.res
            elif res2:
                preds=preds[0]

            preds=torch.sigmoid(preds)
            preds=(preds>0.5).float()

            num_correct+=(preds==y).sum()
            num_pixels+=torch.numel(preds)
            if (preds+y).sum()==0:
                continue
            else:
                dice_score+=(2*(preds*y).sum())/((preds+y).sum())
                count+=1

    print(f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}')
    print('Dice_score:',dice_score/count)
    model.train()
    return (dice_score+1e-6)/(count+1e-6)

def check_accury_total(loader,model1,model2,device='cuda'):
    num_correct=0
    num_pixels=0
    dice_score=0
    model1.eval()
    model2.eval()
    loop=tqdm(loader)
    with torch.no_grad():
        for x,y in loop:
            x=x.type(torch.FloatTensor).to(device)
            y=y.to(device).unsqueeze(1)
            pred1=torch.sigmoid(model1(x))

            preds = torch.sigmoid(model2(pred1))
            preds=(preds>0.5).float()

            num_correct+=(preds==y).sum()
            num_pixels+=torch.numel(preds)
            dice_score+=(2*(preds*y).sum())/((preds+y).sum()+1e-6)

    print(f'Got {num_correct}/{num_pixels} with accuracy {num_correct/num_pixels*100:.2f}')
    print('Dice_score:',dice_score/len(loader))
    model1.train()
    model2.train()
    return dice_score/len(loader)


def save_predictions_as_imgs(loader,model,folder='save_imgs/',res=False,pre=True):
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('creating path {}'.format(folder))
    model.eval()
    for idx,(x_,y_) in enumerate(loader):
        n=x_.size(0)
        for i in range(n):
            x=x_[i].type(torch.FloatTensor).unsqueeze(0).to('cuda')
            y=y_[i].unsqueeze(0).cuda()
            with torch.no_grad():
                preds=model(x)
                if res:
                    preds=preds[-1]

                preds=torch.sigmoid(preds)
                if pre:
                    preds=(preds>0.5).float()
            torchvision.utils.save_image(preds,f'{folder}/{idx}_pred.png')
            torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{idx}.png')
    model.train()


def save_predictions_as_imgs2(loader,model,folder='save_imgs/'):
    model.eval()
    for idx,(x_,y_) in enumerate(loader):
        n=x_.size(0)
        for i in range(n):
            x=x_[i].type(torch.FloatTensor).unsqueeze(0).to('cuda')
            y=y_[i].unsqueeze(0).cuda()
            with torch.no_grad():
                outs=model(x,train=True)
                preds,edge=outs
                preds=torch.sigmoid(model(x))
                preds=(preds>0.5).float()
                # edge=model.edge_net.res
                edge=(torch.sigmoid(edge)>0.5).float()
            torchvision.utils.save_image(
                preds,f'{folder}/{idx}_pred.png'
            )
            torchvision.utils.save_image(edge, f'{folder}/{idx}_edge.png')
            torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{idx}.png')
    model.train()

def save_predictions_as_imgs3(loader,model,folder='save_imgs/'):
    model.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.type(torch.FloatTensor).to('cuda')
        with torch.no_grad():
            outs=model(x,train=True)
            preds,edge=outs
            preds=torch.sigmoid(preds)
            preds=(preds>0.5).float()
            edge=(torch.sigmoid(edge)>0.5).float()
        torchvision.utils.save_image(
            preds,f'{folder}/{idx}_pred.png'
        )
        torchvision.utils.save_image(edge, f'{folder}/{idx}_edge.png')
        torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{idx}.png')
    model.train()

def save_predictions_as_imgs4(loader,model,folder='save_imgs/'):
    model.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.type(torch.FloatTensor).to('cuda')
        with torch.no_grad():
            outs=model(x,res3=True)
            preds,edge=outs
            preds=torch.sigmoid(model(x))
            preds=(preds>0.5).float()
            edge=(torch.sigmoid(edge)>0.5).float()
        torchvision.utils.save_image(
            preds,f'{folder}/{idx}_pred.png'
        )
        torchvision.utils.save_image(edge, f'{folder}/{idx}_edge.png')
        torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{idx}.png')
    model.train()

def save_predictions_as_imgs5(loader,model,folder='save_imgs/',train=False):
    model.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.type(torch.FloatTensor).to('cuda')
        with torch.no_grad():

            # (x5,x4,x3,x2,x1,out)=outs
            if train==True:
                outs = model(x,train)
                l=len(outs)
                for i in range(l):
                    preds=torch.sigmoid(outs[i])
                    preds=(preds>0.5).float()
                    torchvision.utils.save_image(preds, f'{folder}/{idx}_{i}.png')
                torchvision.utils.save_image(y.unsqueeze(1), f'{folder}/{idx}_mask.png')
            else:
                outs=model(x)
                torchvision.utils.save_image(outs, f'{folder}/{idx}_pred.png')
                torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{idx}.png')
    model.train()

def save_predictions_as_imgs6(loader,model,folder='save_imgs/'):
    model.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.type(torch.FloatTensor).to('cuda')
        with torch.no_grad():
            out,edge=model(x,train=True)
            # (x5,x4,x3,x2,x1,out)=outs

            pred=torch.sigmoid(out)
            pred=(pred>0.5).float()
            torchvision.utils.save_image(pred, f'{folder}/{idx}_pred.png')
            edge_=torch.sigmoid(edge)
            edge_=(edge_>0.5).float()
            torchvision.utils.save_image(edge_,f'{folder}/{idx}_edge.png')
            torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{idx}.png')
    model.train()

def save_predictions_as_imgs7(loader,model,folder='save_imgs/'):
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('creating path {}'.format(folder))
    model.eval()
    c=0
    for idx,(xs,ys) in enumerate(loader):
        for i in range(xs.size(0)):
            x=xs[i].type(torch.FloatTensor).to('cuda').unsqueeze(0)
            y= ys[i].unsqueeze(0)
            with torch.no_grad():
                out,edges=model(x,train=True)
                out=(torch.sigmoid(out)>0.5).float()
                torchvision.utils.save_image(out, f'{folder}/{c}_0pred.png')
                l=len(edges)
                # for i in range(l):
                #     preds=torch.sigmoid(edges[i])
                #     preds=(preds>0.5).float()
                #     torchvision.utils.save_image(preds, f'{folder}/{idx}_edge_{i}.png')
                out = (torch.sigmoid(edges) > 0.5).float()
                torchvision.utils.save_image(out, f'{folder}/{c}_edge.png')

                torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{c}.png')
                c+=1
    model.train()


def save_predictions_as_imgs8(loader,model,folder='save_imgs/',res1=False,res2=False,res3=False,softmax=False):
    if not os.path.exists(folder):
        os.mkdir(folder)
    print('saving--------',res2)
    model.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.type(torch.FloatTensor).to('cuda')
        with torch.no_grad():
            if res1:
                out,base_out,edges=model(x,train=True)
                out = (torch.sigmoid(out) > 0.5).float()
                torchvision.utils.save_image(out, f'{folder}/{idx}_0pred.png')
                if res3:
                    l = len(base_out)
                    for i in range(l):
                        a = (torch.sigmoid(base_out[i])).float()
                        torchvision.utils.save_image(a, f'{folder}/{idx}_body{i}.png')
                else:
                    base = (torch.sigmoid(base_out) > 0.5).float()
                    torchvision.utils.save_image(base, f'{folder}/{idx}_base.png')
                    torchvision.utils.save_image(y.unsqueeze(1), f'{folder}/{idx}.png')
                if res2:
                    l=len(edges)
                    for i in range(l):
                        a=(torch.sigmoid(edges[i])).float()
                        torchvision.utils.save_image(a, f'{folder}/{idx}_edge{i}.png')
                else:
                    torchvision.utils.save_image(edges, f'{folder}/{idx}_edge.png')


            else:
                out,edges=model(x,train=True)
                if softmax:
                    out=torch.argmax(torch.softmax(out,dim=1),dim=1)
                    edges=torch.argmax(torch.softmax(edges,dim=1),dim=1)
                out=(torch.sigmoid(out)>0.5).float()
                torchvision.utils.save_image(out, f'{folder}/{idx}_0pred.png')
                # for i,final in enumerate(edges):
                preds=torch.sigmoid(edges)
                preds=(preds>0.5).float()
                torchvision.utils.save_image(preds, f'{folder}/{idx}_edge.png')
                torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{idx}.png')
    model.train()

def save_predictions_as_imgs_total(loader,model1,model2,folder='save_imgs/'):
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('creating path {}'.format(folder))
    model1.eval()
    model2.eval()
    for idx,(x,y) in enumerate(loader):
        x=x.type(torch.FloatTensor).to('cuda')
        with torch.no_grad():
            pred1=model1(x)
            pred1=torch.sigmoid(pred1)
            out,edges=model2(pred1,train=True)
            out=(torch.sigmoid(out)>0.5).float()
            torchvision.utils.save_image(out, f'{folder}/{idx}_0pred.png')
            l=len(edges)
            for i in range(l):
                preds=torch.sigmoid(edges[i])
                preds=(preds>0.5).float()
                torchvision.utils.save_image(preds, f'{folder}/{idx}_edge_{i}.png')
            # out = (torch.sigmoid(edges) > 0.5).float()
            # torchvision.utils.save_image(out, f'{folder}/{idx}_edge.png')

            torchvision.utils.save_image(y.unsqueeze(1),f'{folder}/{idx}.png')
    model1.train()
    model2.train()

def save_predictions_as_imgs_roi(loader,model,folder='save_imgs/',folder2='save_label',folder3='save_edge'):
    if not os.path.exists(folder):
        os.mkdir(folder)
        print('creating path {}'.format(folder))
    if not os.path.exists(folder2):
        os.mkdir(folder2)
        print('creating path {}'.format(folder2))
    model.eval()
    for idx,(x_,y_) in enumerate(loader):
        n=x_.size(0)
        for i in range(n):
            x=x_[i].type(torch.FloatTensor).unsqueeze(0).to('cuda')
            y=y_[i].unsqueeze(0).cuda()
            with torch.no_grad():
                preds=model(x)
                preds=torch.sigmoid(preds)

            torchvision.utils.save_image(preds,f'{folder}/{idx}_roi.png')
            torchvision.utils.save_image(y.unsqueeze(1),f'{folder2}/{idx}.png')
    model.train()
    save_edge(folder2,folder3)