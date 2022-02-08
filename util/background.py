import nibabel as nib
import numpy as np
from nilearn.image.image import _crop_img_to as crop_img_to  ,as_ndarray
file = 'D:/zhiwensuo/data_Naso_GTV/1/label.nii.gz'
image = nib.load(file)
print(image.shape)
import matplotlib.pyplot as plt
data = image.get_fdata()[:,:,52]
plt.imshow(data)
plt.show()
a=image.get_fdata()
print(a.shape,type(a))
# def get_slice_index(data, rtol=1e-8):
#     infinity_norm = max(-data.min(), data.max())
#     passes_threshold = np.logical_or(data < -rtol * infinity_norm,
#                                      data > rtol * infinity_norm)  ##
#     if data.ndim == 4:
#         passes_threshold = np.any(passes_threshold, axis=-1)
#
#     coords = np.array(np.where(passes_threshold))
#     start = coords.min(axis=1)
#     end = coords.max(axis=1) + 1
#
#     # pad with one voxel to avoid resampling problems
#     start = np.maximum(start - 1, 0)
#     end = np.minimum(end + 1, data.shape[:3])
#
#     slices = [slice(s, e) for s, e in zip(start, end)]
#     return slices
# def have_back(image):
#     background_value=0
#     tolerance=0.00001
#     is_foreground = np.logical_or(image.get_fdata() < (background_value - tolerance),
#                                   image.get_fdata()> (background_value + tolerance))
#     foreground = np.zeros(is_foreground.shape, dtype=np.uint8)
#     foreground[is_foreground] = 1
#     return foreground
# foreground = have_back(image)
# crop = get_slice_index(foreground)
# print(crop)
# image_o = crop_img_to(image, crop, copy=True)
# import matplotlib.pyplot as plt
# data_o = image_o.get_fdata()[:,:,15]
# plt.imshow(data_o)
# plt.show()
# print(image_o.shape)
# print('111')
#

def get_none_zero_region(im, margin):
    """
    get the bounding box of the non-zero region of an ND volume
    """
    input_shape = im.shape
    if(type(margin) is int ):
        margin = [margin]*len(input_shape)
    assert(len(input_shape) == len(margin))
    indxes = np.nonzero(im)
    idx_min = []
    idx_max = []

    for i in range(len(input_shape)):
        idx_min.append(indxes[i].min())
        idx_max.append(indxes[i].max())

    for i in range(len(input_shape)):
        idx_min[i] = max(idx_min[i] - margin[i], 0)
        idx_max[i] = min(idx_max[i] + margin[i], input_shape[i] - 1)
    return idx_min, idx_max

def crop_ND_volume_with_bounding_box(volume, min_idx, max_idx):
    """
    crop/extract a subregion form an nd image.
    """
    dim = len(volume.shape)
    assert(dim >= 2 and dim <= 5)
    if(dim == 2):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1))]
    elif(dim == 3):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1))]
    elif(dim == 4):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1))]
    elif(dim == 5):
        output = volume[np.ix_(range(min_idx[0], max_idx[0] + 1),
                               range(min_idx[1], max_idx[1] + 1),
                               range(min_idx[2], max_idx[2] + 1),
                               range(min_idx[3], max_idx[3] + 1),
                               range(min_idx[4], max_idx[4] + 1))]
    else:
        raise ValueError("the dimension number shoud be 2 to 5")
    return output


margin = 5
bbmin, bbmax = get_none_zero_region(a, margin)
print(bbmin, bbmax)
volume = crop_ND_volume_with_bounding_box(a, bbmin, bbmax)
print(volume.shape)
data_o = a[:,:,52]
plt.imshow(data_o)
plt.show()
print('111')



