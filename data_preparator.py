import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pydicom
import os


def prepare_data():
    train_data_path = os.path.abspath('train_data/')
    num_of_images = len(os.listdir(train_data_path))
    #load_first image to setup size
    vertebra = pydicom.dcmread(os.path.join(train_data_path, os.listdir(train_data_path)[0]))
    y = vertebra.pixel_array.shape[0]
    x = vertebra.pixel_array.shape[1]
    train_imgs = np.ndarray((num_of_images, y, x), dtype=np.uint16)
    print('Size of images: {0} x {1}'.format(y, x))

    print('Number of images for training purposes: ' + str(num_of_images))
    print('-'*40)
    i = 0
    for image in sorted(os.listdir(train_data_path)):
        vertebra = pydicom.dcmread(os.path.join(train_data_path, image))
        img = np.array(vertebra.pixel_array)
        minimum = np.min(vertebra.pixel_array)
        if minimum < 0:
            img = img + abs(minimum)
        train_imgs[i] = img
        i += 1
        print('Prepared images: {0} of {1} '.format(i, num_of_images))
    print('-'*40)
    np.save('train_images.npy', train_imgs)


def load_train_data():
    return np.load('train_images.npy')


def prepare_mask():
    mask_path = os.path.abspath('mask/')
    num_of_masks = len(os.listdir(mask_path))
    # load_first image to setup size
    first_mask = mpimg.imread(os.path.join(mask_path, os.listdir(mask_path)[0]))
    y = first_mask.shape[0]
    x = first_mask.shape[1]
    masks = np.ndarray((num_of_masks, y, x), dtype=np.uint8)
    print('Number of mask for training purposes: ' + str(num_of_masks))
    print('-'*40)
    i = 0
    for image in sorted(os.listdir(mask_path)):
        mask = mpimg.imread(os.path.join(mask_path, image))
        mask[mask < 1] = 0
        masks[i] = mask
        i += 1
        print('Prepared masks: {0} of {1} '.format(i, num_of_masks))
    print('-' * 40)
    np.save('masks.npy', masks)

def load_masks():
    return np.load('masks.npy')


if __name__ == '__main__':
    prepare_data()
    prepare_mask()

