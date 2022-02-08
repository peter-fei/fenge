from PIL import Image
import numpy as np

a=np.array(Image.open('../data_Naso_crop1/train/image/1_85.png'))
b=np.array(Image.open('../data_Naso_crop1/train/label/1_85_label.png'))
h=np.where(b==255)
# print(np.max(b),np.min(b))
# print(h)
# print(a.shape,b.shape)
# c=a+b
# d=Image.fromarray(c)
Image._show(Image.fromarray(a))
Image._show(Image.fromarray(b))
# Image._show(d)
# c=a+b
# print(np.max(c))
a[b==255]=255
Image._show(Image.fromarray(a))