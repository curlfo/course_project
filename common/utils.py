import nibabel as nib
import os
from scipy.ndimage import zoom
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import train.augmentation as augmentation
import pydicom
from skimage.transform import resize
import keras


def show_slices(slices):
    fig, axes = plt.subplots(1, len(slices), figsize=(10, 4))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")


def show_img(img_data, header="Center slices for EPI image"):
    slice_0 = img_data[img_data.shape[0] // 2, :]
    slice_1 = img_data[:, img_data.shape[1] // 2]
    slice_2 = img_data[:, :, img_data.shape[2] // 2]
    show_slices([slice_0, slice_1, slice_2])
    plt.suptitle(header)
    plt.show()
    plt.close()


def load_2d_img_as_ndarray(dicom_path, target_shape=(256, 256)):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    im = resize(img, target_shape)
    im[im < -1000] = -1000
    im[im > 500] = 500
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im


def load_img_as_ndarray(nifti_path, target_shape=(256, 256, 128)):
    """
      Loads nifti image as ndarray, normalizes it to 0-1 and reshapes to target size.

      :param nifti_path: path to nifti image
      :param target_shape: target shape

      :return: image as ndarray of target shape, normalized to 0-1
    """
    img = nib.load(nifti_path)
    im = img.get_data()
    im = zoom(im.astype(float), (1.0 * target_shape[_] / im.shape[_] for _ in range(len(target_shape))))
    im[im < -1000] = -1000
    im[im > 500] = 500
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im


def load_img_and_mask_as_ndarray(nifti_path, mask_path, debug=False):
    """
         Loads nifti image and mask as ndarray, normalizes it to 0-1.
         During loading image is cropped by mask.

         :param nifti_path: path to nifti image
         :param mask_path: path to mask file (nifti format)
         :param debug: flag for debug output

         :return:
            masked cropped image as ndarray, normalized to 0-1 (out of mask voxels are 0)
            cropped mask (same shape as image)
        """
    img = nib.load(nifti_path)
    if debug:
        print('affine', img.affine)
    im = img.get_data()
    mask = nib.load(mask_path)
    mask_data = mask.get_data()
    mask_data[mask_data > 1] = 1
    bounds = []
    proj_dims = [(1, 2), (0, 2), (0, 1)]
    for dim in range(len(im.shape)):
        projection = np.sum(mask_data, axis=proj_dims[dim]).flatten()
        idx = np.argwhere(projection > 0)
        first = int(max(0, idx[0]))
        last = int(min(len(projection), idx[-1] + 1))
        bounds.append((first, last))
    if debug:
        croped_shape = (bounds[0][1] - bounds[0][0], bounds[1][1] - bounds[1][0], bounds[2][1] - bounds[2][0])
        print('initial shape', im.shape)
        print('mask shape', mask_data.shape)
        print('cropped shape', croped_shape)
    im = im[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]]
    return im, mask_data[bounds[0][0]:bounds[0][1], bounds[1][0]:bounds[1][1], bounds[2][0]:bounds[2][1]], img.affine


def extract_one_lung(left, image, mask, target_shape, augment=False, debug=False):
    if left:
        image = image[:image.shape[0] // 2, :, :]
        mask = mask[:mask.shape[0] // 2, :, :]
    else:
        image = image[image.shape[0] // 2:, :, :][::-1, :, :]
        mask = mask[mask.shape[0] // 2:, :, :][::-1, :, :]

    if augment:
        image, mask = augmentation.augment_image_masked_3d(image, mask)

    image = zoom(image.astype(float), (1.0 * target_shape[_] / image.shape[_] for _ in range(len(target_shape))))
    mask = zoom(mask.astype(float), (1.0 * target_shape[_] / mask.shape[_] for _ in range(len(target_shape))))
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0

    if debug:
        show_img(image)

    return image, mask


def get_all_CT_filenames_to_load(dir_path):
    return glob(os.path.join(dir_path, "*.nii*"))


def get_mask_path_for_CT_filenames(ct_filepathes, mask_dir):
    """
         Matches CT image with its masks by filename.

         :param ct_filepathes: list or image pathes
         :param mask_dir: root directory of mask files

         :return:
            list of mask pathes
       """
    mask_pathes = []
    for p in ct_filepathes:
        mp = os.path.join(mask_dir, os.path.basename(p))
        assert os.path.exists(mp)
        mask_pathes.append(mp)
    return mask_pathes
