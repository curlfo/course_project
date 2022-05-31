import numpy as np
import tensorflow.keras as keras
import common.utils as utils
import train.augmentation as augmentation
from scipy.ndimage import zoom


class Data3DGeneratorAEC(keras.utils.Sequence):
    def __init__(self, list_filepathes, augment=True, batch_size=32, dim=(256, 256, 128, 1), shuffle=True):
        self.dim = dim
        self.augment = augment
        self.batch_size = batch_size
        self.list_filepathes = list_filepathes
        self.shuffle = shuffle
        self.images = []
        for i in range(len(self.list_filepathes)):
            print(i, self.list_filepathes[i])
            try:
                self.images.append(utils.load_img_as_ndarray(self.list_filepathes[i], self.dim[:3]))
            except:
                print("ERROR!")
        self.X = np.empty(np.concatenate(([self.batch_size], self.dim)))
        self.on_epoch_end()

    def __len__(self):
        # return int(np.floor(len(self.list_filepathes) / self.batch_size))
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(indexes)

        return X, X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        for i, ID in enumerate(indexes):
            # image = utils.load_img_as_ndarray(self.list_filepathes[ID], self.dim[:3])
            image = self.images[ID]
            # if self.augment:
            #    image = augmentation.augment_image_3d(image)
            self.X[i,] = image.reshape(np.concatenate((image.shape, [1])))

        return self.X


class Data2DGeneratorAEC(keras.utils.Sequence):
    def __init__(self, list_filepathes, augment=True, batch_size=32, dim=(256, 256, 1), shuffle=True):
        self.dim = dim
        self.augment = augment
        self.batch_size = batch_size
        self.list_filepathes = list_filepathes
        self.shuffle = shuffle
        self.images = []
        for i in range(len(self.list_filepathes)):
            print(i, self.list_filepathes[i])
            try:
                self.images.append(utils.load_2d_img_as_ndarray(self.list_filepathes[i], self.dim[0:2]))
            except:
                print("ERROR!")
        self.X = np.empty(np.concatenate(([self.batch_size], self.dim)))
        self.on_epoch_end()

    def __len__(self):
        # return int(np.floor(len(self.list_filepathes) / self.batch_size))
        return int(np.floor(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(indexes)

        return X, X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        for i, ID in enumerate(indexes):
            # image = utils.load_img_as_ndarray(self.list_filepathes[ID], self.dim[:3])
            image = self.images[ID]
            # if self.augment:
            #    image = augmentation.augment_image_3d(image)
            self.X[i,] = image.reshape(np.concatenate((image.shape, [1])))

        return self.X


class PerLungData3DGeneratorAEC(keras.utils.Sequence):
    def __init__(self, list_filepathes, list_masks, augment=True, batch_size=32, dim=(256, 256, 128, 1), shuffle=True):
        self.dim = dim
        self.augment = augment
        self.batch_size = batch_size
        self.list_filepathes = list_filepathes
        self.list_masks = list_masks
        assert len(list_filepathes) == len(list_masks), (list_filepathes, list_masks)
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(2 * len(self.list_filepathes) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(indexes)

        return X, X

    def on_epoch_end(self):
        self.indexes = np.arange(2 * len(self.list_filepathes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        X = np.empty(np.concatenate(([self.batch_size], self.dim)))

        for i, ID in enumerate(indexes):
            target_shape = self.dim[:3]

            image, mask = utils.load_img_and_mask_as_ndarray(self.list_filepathes[ID // 2], self.list_masks[ID // 2])

            X[i,] = utils.extract_one_lung(ID % 2 == 0, image, mask, target_shape, self.augment, False)

        return X


class OneLungData3DGeneratorAEC(keras.utils.Sequence):
    def __init__(self, list_filepathes, list_masks, left, augment=True, batch_size=32, dim=(256, 256, 128, 1),
                 shuffle=True):
        self.dim = dim
        self.augment = augment
        self.batch_size = batch_size
        self.list_filepathes = list_filepathes
        self.list_masks = list_masks
        assert len(list_filepathes) == len(list_masks), (list_filepathes, list_masks)
        self.shuffle = shuffle
        self.left = left
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_filepathes) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        X = self.__data_generation(indexes)

        return X, X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_filepathes))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)

        X = np.empty(np.concatenate(([self.batch_size], self.dim)))

        for i, ID in enumerate(indexes):
            target_shape = self.dim[:3]

            image, mask = utils.load_img_and_mask_as_ndarray(self.list_filepathes[ID],
                                                             self.list_masks[ID])

            X[i,] = utils.extract_one_lung(self.left, image, mask, target_shape, self.augment, False)

        return X
