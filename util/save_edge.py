import os
import cv2 as cv
import numpy as np


def save_edge(base_path,save_path):
    files = os.listdir(base_path)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
        print('creating path {}'.format(save_path))
    num = 0
    for file in files:
        file_path = os.path.join(base_path, file)
        img = cv.imread(file_path)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        contours, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, 1)
        img2 = np.ones(img_gray.shape)
        cv.drawContours(img2, contours, -1, (255, 255, 255), 2)
        save_img = file.replace('.png', '_edge.png')
        cv.imwrite(os.path.join(save_path, save_img), img2)
        num += 1
    print(f'done with num {num}')

if __name__=='__main__':

    # base_path= 'D:/zhiwensuo/pytorch/project/Naso-gtv-data/train/label'
    # save_path='D:/zhiwensuo/pytorch/project/Naso-gtv-data/train/edge2'
    # base_path='D:/zhiwensuo/pytorch/U-net-master/dataset/test/label'
    # save_path='D:/zhiwensuo/pytorch/U-net-master/dataset/test/edge_black'
    base_path = 'D:/zhiwensuo/pytorch/project/data_Naso_crop2/train/label'
    save_path = 'D:/zhiwensuo/pytorch/project/data_Naso_crop2/train/edge'
    save_edge(base_path,save_path)
    # files=os.listdir(base_path)
    # num=0
    # for file in files:
    #     file_path=os.path.join(base_path,file)
    #     img=cv.imread(file_path)
    #     img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #     contours, thresh = cv.threshold(img_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    #     contours, hierarchy = cv.findContours(thresh, cv.RETR_LIST, 1)
    #     img2=np.ones(img_gray.shape)
    #     cv.drawContours(img2, contours, -1, (255, 255, 255),2)
    #     save_img=file.replace('.png','_edge.png')
    #     cv.imwrite(os.path.join(save_path,save_img),img2)
    #     num+=1
    # print(f'done with num {num}')