import numpy
import cv2
import os

label_path='D:/zhiwensuo/pytorch/project/data_Naso2/test/label_test'
img_path='D:/zhiwensuo/pytorch/project/data_Naso2/test/image_test'
save_path1='D:/zhiwensuo/pytorch/project/data_Naso2/test/label'
save_path2='D:/zhiwensuo/pytorch/project/data_Naso2/test/image'

def find_files(base_path,img_path):
    files=os.listdir(base_path)
    for file in files:
        path = os.path.join(base_path, file)
        if os.path.isdir(path):
            _path = os.path.join(img_path, file)
            find_files(path,_path)
        else:
            _path = os.path.join(img_path, file[5:])
            img=cv2.imread(path)
            print(_path)
            img2=cv2.imread(_path)
            if (img>0).sum()>0:
                cv2.imwrite((os.path.join(save_path1,path[-8:]).replace('.png','_edge.png')),img)
                print(os.path.join(save_path2, _path[-8:]))
                cv2.imwrite((os.path.join(save_path2, _path[-8:])), img2)

def save_nozero_imgs(img_path,label_path,save_img_path,save_label_path):
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)
        print(f'making{save_img_path}')
    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)
        print(f'making{save_label_path}')
    for imgfile in os.listdir(img_path):
        labelfile=imgfile.replace('.png','_label.png')
        img=cv2.imread(os.path.join(img_path,imgfile))
        label=cv2.imread(os.path.join(label_path,labelfile))
        # print(imgfile,labelfile)
        # print(label)
        if (label>0).sum()>0:
            cv2.imwrite(os.path.join(save_img_path,imgfile),img)
            cv2.imwrite(os.path.join(save_label_path,labelfile),label)
    print('done')
# find_files(label_path,img_path)
# print(0)
img_path='D:/zhiwensuo/data_Naso/test/image'
label_path='D:/zhiwensuo/data_Naso/test/label'
save_img_path='D:/zhiwensuo/pytorch/project/data_Naso/test/image'
save_label_path='D:/zhiwensuo/pytorch/project/data_Naso/test/label'
save_nozero_imgs(img_path,label_path,save_img_path,save_label_path)