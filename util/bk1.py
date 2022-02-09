import os

import PIL.Image
import numpy as np
from util.save_edge import *
from skimage.filters import threshold_otsu
from skimage import measure, exposure
import SimpleITK as sitk
import skimage

def tight_crop(img, size=None):
    img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:])+1

    img_circ = (img_label == largest_label)
    img_xs = np.sum(img_circ, 0)
    img_ys = np.sum(img_circ, 1)
    xs = np.where(img_xs>0)
    ys = np.where(img_ys>0)
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    img_crop = img[y_lo:y_hi, x_lo:x_hi, :]

    return img_crop

import scipy.misc
from PIL import Image
import matplotlib.pyplot as plt

#
# img = np.array(Image.open('../data_Naso/train/image/1_75.png'))
# img_crop = tight_crop(img)
# plt.imshow(img)
# plt.show()
# plt.imshow(img_crop)
# plt.show()
# PIL.Image.Image.show(img_crop)
# pilImage = Image.fromarray(skimage.util.img_as_ubyte(img_crop))
# pilImage.show()
# print(img_crop.shape)
import cv2

def get_crop(file):
    img = cv2.imread(file)
    if img.sum()==0:
        return (9999,0,9999,0)
    img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:]) + 1
    # print(img_bw)
    img_bw=(img_bw*255).astype(np.uint8)
    # print(img_bw)
    # img_gray[img_bw==False]=0
    # cv2.imshow('da',img_bw)
    # cv2.waitKey(0)
    img_circ = (img_label == largest_label)
    img_xs = np.sum(img_circ, 0)
    img_ys = np.sum(img_circ, 1)
    xs = np.where(img_xs > 0)
    ys = np.where(img_ys > 0)
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    return (y_lo,y_hi,x_lo,x_hi)

def get_crop3D1(file):
    # norm_factor=200
    img_obj = sitk.ReadImage(file)
    img_grays = sitk.GetArrayFromImage(img_obj)
    img_data_clipped = np.clip(img_grays, -125, 300)
    img_grays = (img_data_clipped - (-125)) / (300 - (-125))
    # img_grays = img_grays / norm_factor
    img_gray=img_grays
    # print(img_gray.shape)
# img = cv2.imread(file)
# img_gray = np.mean(img, 2)
    img_bw = img_gray > threshold_otsu(img_gray)
    img_label = measure.label(img_bw, background=0)
    largest_label = np.argmax(np.bincount(img_label.flatten())[1:]) + 1
    # print(img_bw)
    img_bw=(img_bw*255).astype(np.uint8)
    # print(img_bw.shape)
    # print(img_bw)
    # img_gray[img_bw==False]=0
    # cv2.imshow('da',img_bw)
    # cv2.waitKey(0)
    img_circ = (img_label == largest_label)
    # print(img_circ.shape,largest_label,img_label.shape)
    img_ys = np.sum(img_circ, (0,2))
    img_xs = np.sum(img_circ, (0,1))
    img_zs = np.sum(img_circ, (1,2))
    # print(img_xs.shape,img_ys.shape,img_zs.shape)
    xs = np.where(img_xs > 0)
    ys = np.where(img_ys > 0)
    zs = np.where(img_zs >0)
    # print(len(xs))
    x_lo = np.min(xs)
    x_hi = np.max(xs)
    y_lo = np.min(ys)
    y_hi = np.max(ys)
    z_lo=np.min(zs)
    z_hi=np.max(zs)

    # print(z_lo,z_hi,y_lo, y_hi, x_lo, x_hi)
    i=img_grays[:,y_lo:y_hi,x_lo:x_hi]
    # print(i.shape,img_grays.shape)
    # cv2.imshow('aa',img_grays[20])
    # cv2.imshow('ss',i[20])
    # cv2.waitKey(0)
    return (z_lo,z_hi,y_lo,y_hi,x_lo,x_hi)
def save_imgs(img_floder,label_floder,save_img_path,save_label_path,l1,l2=None):
    # print(l1,l2)

    for i,file in enumerate(os.listdir(img_floder)):
        y_min, y_max, x_min, x_max = l1
        filename=os.path.join(img_floder,file)
        labelname=os.path.join(label_floder,file.replace('.png','_label.png'))
        # labelname=os.path.join(label_floder,file)
        # print(filename)
        img=cv2.imread(filename)
        label=cv2.imread(labelname)
        img_crop = img[x_min:x_max, y_min:y_max]
        label_crop=label[x_min:x_max, y_min:y_max]

        if l2 is not None:
            y_min, y_max, x_min, x_max = l2
            img_crop = img_crop[x_min:x_max, y_min:y_max]
            label_crop = label_crop[x_min:x_max, y_min:y_max]

        save_img=os.path.join(save_img_path,file)
        save_label=os.path.join(save_label_path,file.replace('.png','_label.png'))
        cv2.imwrite(save_img,img_crop)
        cv2.imwrite(save_label, label_crop)
    print(l)
if __name__=='__main__':
    # 3D
    # img_floder='D:/zhiwensuo/data_Naso_3D/train'
    # label_floder='D:/zhiwensuo/data_Naso_3D/label'
    # save_img_path='D:/zhiwensuo/data_Naso_3D_crop/train'
    # save_label_path ='D:/zhiwensuo/data_Naso_3D_crop/train'
    # z_min,z_max,y_min, y_max, x_min, x_max = (9999,0,9999, 0, 9999, 0)
    # for i,file in enumerate(os.listdir(img_floder)):
    #     filename=os.path.join(img_floder,file)
    #     a=get_crop3D1(filename)
    #     # print(a)
    #     z_min_, z_max_, y_min_, y_max_, x_min_, x_max_=a
    #     y_min=min(y_min,y_min_)
    #     y_max=max(y_max,y_max_)
    #     x_min=min(x_min,x_min_)
    #     x_max=max(x_max,x_max_)
    #     z_min = min(z_min, z_min_)
    #     z_max = max(z_max, z_max_)
    # l=(z_min,z_max,y_min,y_max,x_min,x_max)
    # save_imgs(train_img_floder, train_label_floder, train_save_img_path, train_save_label_path, l)
    # print(l)
    # a=get_crop3D1('D:/zhiwensuo/data_Naso_3D/train/1data.nii.gz')
    # print(a)

    # 2D
    y_min,y_max,x_min,x_max = (9999,0,9999, 0)
    img_floder='../data_Naso_crop1/train/label'
    # img_floder = '../save_imgs_bi_5'
    # label_floder='../preprocess-npz-data/test/label'
    # save_img_path='D:/zhiwensuo/pytorch/project/data_Naso_crop2/test/image'
    # save_label_path ='../data_Naso_crop2/test/label'

    train_img_floder = '../data_Naso_total/test/image'
    train_label_floder = '../data_Naso_total/test/label'
    train_save_img_path = '../data_Naso_total_crop/test/image'
    train_save_label_path = '../data_Naso_total_crop/test/label'
    # train_save_edge_path='../data_Naso_total_crop/train/edge'

    # if not os.path.exists(save_img_path):
    #     os.makedirs(save_img_path)
    #     print(f'making path{save_img_path}')
    # if not os.path.exists(save_label_path):
    #     os.makedirs(save_label_path)
    #     print(f'making path{save_label_path}')
    if not os.path.exists(train_save_img_path):
        os.makedirs(train_save_img_path)
        print(f'making path{train_save_img_path}')
    if not os.path.exists(train_save_label_path):
        os.makedirs(train_save_label_path)
        print(f'making path{train_save_label_path}')
    # print(os.listdir(img_floder))
    #
    # for i,file in enumerate(os.listdir(img_floder)):
    #     filename=os.path.join(img_floder,file)
    #     # print(filename)
    #     y_min_,y_max_,x_min_,x_max_=get_crop(filename)
    #     y_min=min(y_min,y_min_)
    #     y_max=max(y_max,y_max_)
    #     x_min=min(x_min,x_min_)
    #     x_max=max(x_max,x_max_)
    #
    #
    # if (y_max - y_min) % 8:
    #     y_max+=16-(y_max-y_min)%16
    # if(x_max-x_min)%8:
    #     x_max+=16-(x_max-x_min)%16
    # print(y_min, y_max, x_min, x_max)
    l=[y_min,y_max,x_min,x_max]
    l1=[142,382,167,359]
    # l1 = [167, 359, 142, 382]
    # l2=[67-16, 195+16, 48-16, 112+16]
    l2=[53-16,149+16,62-16,142+16]
    # l2=[48,112,67,195]
    save_imgs(train_img_floder, train_label_floder, train_save_img_path, train_save_label_path, l1,l2)
    # save_edge(train_save_label_path, train_save_edge_path)
    # print(l)

    a=cv2.imread('../data_Naso_crop1/train/label/2_78_label.png')
    b=cv2.imread('../data_Naso_total/train/label/2_78_label.png')
    y_min, y_max, x_min, x_max = l1
    b=b[x_min:x_max, y_min:y_max]
    y_min, y_max, x_min, x_max = l2
    cv2.imshow('qqq',a[x_min:x_max,y_min:y_max])
    cv2.imshow('qq2', b[x_min:x_max, y_min:y_max])
    cv2.imshow('q1q', b)
    cv2.imshow('q11', a)

    cv2.waitKey(0)
    # for i,file in enumerate(os.listdir(train_img_floder)):
    #     filename=os.path.join(train_img_floder,file)
    #     y_min_,y_max_,x_min_,x_max_=get_crop(filename)
    #     y_min=min(y_min,y_min_)
    #     y_max=max(y_max,y_max_)
    #     x_min=min(x_min,x_min_)
    #     x_max=max(x_max,x_max_)
    #
    # if (y_max-y_min)%8:
    #     y_max+=16-(y_max-y_min)%16
    # if(x_max-x_min)%8:
    #     x_max+=16-(x_max-x_min)%16
    # l=[y_min,y_max,x_min,x_max]
    # save_imgs(img_floder,label_floder,save_img_path,save_label_path,l1,l2)



    # for i,file in enumerate(os.listdir(img_floder)):
    #     filename=os.path.join(img_floder,file)
    #     labelname=os.path.join(label_floder,file.replace('.png','_label.png'))
    #     # print(filename)
    #     img=cv2.imread(filename)
    #     label=cv2.imread(labelname)
    #     img_crop = img[y_min:y_max, x_min:x_max, :]
    #     label_crop=label[y_min:y_max, x_min:x_max, :]
    #     save_img=os.path.join(save_img_path,file)
    #     save_label=os.path.join(save_label_path,file.replace('.png','_label.png'))
    #     cv2.imwrite(save_img,img_crop)
    #     cv2.imwrite(save_label, label_crop)
    # print(l)