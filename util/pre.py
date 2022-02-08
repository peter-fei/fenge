MIN_BOUND = -1000.0
MAX_BOUND = 400.0

import cv2
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
# import h5py
import os
from PIL import Image
import time

# img = nib.load(r"../../data_Naso_GTV/27/data.nii.gz")
# label = nib.load(r"../../data_Naso_GTV/1/label.nii.gz")
# base_path=r'../../data_Naso_GTV'
# files=(os.listdir(base_path))
# Convert them to numpy format,
#
# for index in files:
#     img_path=os.path.join(base_path,index,'data.nii.gz')
#     label_path=os.path.join(base_path,index,'label.nii.gz')
#     img = nib.load(img_path)
#     label=nib.load(label_path)
#     img_data=img.get_fdata()
#     label_data=label.get_fdata()
#     img_data_clipped = np.clip(img_data, -125, 275)
#     img_data_normalised = (img_data_clipped - (-125)) / (275 - (-125))
#     for i in range(img_data_clipped.shape[2]):
#         formattedi = "{:03d}".format(i)
#         slice000 = img_data_normalised[:, :, i] * 255
#         np.savetxt(r"label.txt", label_data[:, :, 6], delimiter=',', fmt='%5s')
#         label_slice000 = label_data[:, :, i] * 255
#
#         # print(slice000.shape, type(slice000))
#
#         image = Image.fromarray(slice000)
#         image = image.convert("RGB")
#
#         label = Image.fromarray(label_slice000)
#         label = label.convert("L")
#         if np.array(label).sum()>0:
#             image.save(f'D:/zhiwensuo/pytorch/project/data_Naso/image/{index}_' + str(i) + ".png")
#             label.save(f'D:/zhiwensuo/pytorch/project/data_Naso/label/{index}_' + str(i) + "_label.png")
#             print(f'{index}_{i}')
#
#
# # print(cont)
#
# data = img.get_fdata()
# # label_data = label.get_fdata()
# print(data.shape)
"""
Convert them to numpy format, clip the images within [-125, 275], normalize each 3D image to [0, 1], 
and extract 2D slices from 3D volume for training cases while keeping the 3D volume in h5 format for testing cases.
"""
# clip the images within [-125, 275],
# data_clipped = np.clip(data, -125, 275)
#
# # normalize each 3D image to [0, 1], and
# data_normalised = (data_clipped - (-125)) / (275 - (-125))

# 喉颈部、鼻咽、咽喉部的窗宽和窗位常设在300 Hu~350 Hu和30 Hu~50 Hu,能满足该部位的解剖和病灶显示,
def norm_img(image): # 归一化像素值到（0，1）之间，且将溢出值取边界值
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

def linear_norm(img,min,max):

    img=(img-min)*255/(max-min)
    img[img>255]=255
    img[img<0]=0
    return img

def gamma_norm(img,min,max,gamma=2):
    img = 255.0 * pow(img / (max - min), 1.0 / gamma);
    img[img > 255] = 255
    img[img < 0] = 0
    return img
if __name__=='__main__':

    window_center=300
    window_width=30
    min = (2 * window_center - window_width) / 2.0 + 0.5
    max = (2 * window_center + window_width) / 2.0 + 0.5
    a=cv2.imread('../data_Naso_crop1/train/image/1_75.png')
    img_path='../../../data_Naso_GTV/1/data.nii.gz'
    label_path='../../../data_Naso_GTV/1/label.nii.gz'
    img = nib.load(img_path)
    img_data = img.get_fdata()
    label = nib.load(label_path)
    label_data = label.get_fdata()
    i1=img_data
    # i1=linear_norm(img_data,min,max)
    print(np.max(img_data),np.min(img_data))
    print(np.max(a),np.min(a))
    print(np.max(i1),np.min(i1))
    img_data_clipped = np.clip(img_data, -100, 200)
    img_data_normalised = (img_data_clipped - (-100)) / (200 - (-100))
    print(img_data_normalised.shape)
    i_=img_data_normalised[:,:,99]
    l_=label_data[:,:,99]
    cv2.imshow('ss',i_)
    cv2.imshow('s1', l_)
    cv2.imshow('s2', i_+l_)
    print(l_.shape)
    cv2.waitKey(0)

    # img_gray =l_
    # img_gray=(l_ * 255).astype(np.uint8)
    # print(np.min(img_gray),np.max(img_gray))
    # l = Image.fromarray(l_)
    # l_ = l.convert("L")
    # img_gray=np.array(l_)
    # i_=cv2.cvtColor((i_*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
    # # img_gray= cv2.cvtColor(img_gray, cv2.COLOR_BGR2GRAY)
    # contours, thresh = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, 1)
    # # img2 = np.zeros(img_gray.shape)
    # draw_img = cv2.merge((thresh.copy(), thresh.copy(), thresh.copy()))
    # cv2.drawContours(draw_img, contours, -1, (255,255,255), 10)
    # # print(contours)
    # cv2.imshow('ss',i_)
    # cv2.imshow('s1', draw_img)
    # cv2.imshow('s2', i_+draw_img)
    # cv2.waitKey(0)
    # plt.imshow(img_data[:,:,1])
    # plt.show()
    # plt.imshow(i1[:,:,1])
    # plt.show()
    # cv2.imwrite('a1.png',i1[:,:,1])
