import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import math
import pydicom


def to_signed(img):
    minimum = np.min(img)
    if minimum < 0:
        img = img + abs(minimum)
    return img


def normalize(img):
    maximum = np.max(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i, j] = img[i, j] * 255 / maximum
    return img.astype(np.int8)


def count_gradient(img, size):
    gradients = np.zeros((img.size, size, size), dtype=np.int8)
    middle = math.trunc(size / 2)
    gradients_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.int8)
    gradients_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.int8)
    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1] - 1):
            gradients_x[i, j] = img[i, j + 1] - img[i, j]
            gradients_y[i, j] = img[i + 1, j] - img[i, j]
    gradients_x = np.pad(gradients_x, middle, mode='constant')
    gradients_y = np.pad(gradients_y, middle, mode='constant')
    print(gradients)
    pixel = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gradients[pixel] = gradients_x[i:i + size, j:j+size]
            pixel = pixel + 1
            #should work
            #try with smaller matrix
            #add gradients_y to gradients

    return gradients_x


def prepare_vectorized_data():
    data_path = os.path.abspath('train_data/')
    num_of_images = len(os.listdir(data_path))
    for img in sorted(os.listdir(data_path)):
        data = pydicom.dcmread(os.path.join(data_path, img)).pixel_array
        # Signed to unsigned
        data = to_signed(data)
        # Normalization of data
        data = normalize(data)
        # Calling each method for preparing features
        features = count_gradient(data, 15)
        print(features)
        plt.imshow(features)
        plt.show()
        # Append images to np.array in vectorized form

    # Save vectorized data and features


def prepare_data():
    data_path = os.path.abspath('train_data/')
    num_of_images = len(os.listdir(data_path))
    # load first image to set size of train_images
    data = pydicom.dcmread(os.path.join(data_path, os.listdir(data_path)[0]))
    y = data.pixel_array.shape[0]
    x = data.pixel_array.shape[1]
    # numpy array for saving all images
    train_images = np.ndarray((num_of_images, y, x), dtype=np.uint16)
    print('Size of images: {0} x {1}'.format(y, x))
    print('-' * 40)
    print('Number of images for training purposes: ' + str(num_of_images))
    print('-' * 40)
    i = 0
    for img in sorted(os.listdir(data_path)):
        data = pydicom.dcmread(os.path.join(data_path, img))
        image = np.array(data.pixel_array)
        # convert images from signed type to unsigned
        minimum = np.min(data.pixel_array)
        if minimum < 0:
            image = image + abs(minimum)
        train_images[i] = image
        i += 1
        print('Prepared images: {0} of {1} '.format(i, num_of_images))
    print('-' * 40)
    np.save('train_images.npy', train_images)


def load_train_data():
    return np.load('train_images.npy')


def prepare_mask():
    mask_path = os.path.abspath('mask/')
    num_of_masks = len(os.listdir(mask_path))
    # load first mask to set size of masks
    first_mask = mpimg.imread(os.path.join(mask_path, os.listdir(mask_path)[0]))
    y = first_mask.shape[0]
    x = first_mask.shape[1]
    # numpy array for saving all masks
    masks = np.ndarray((num_of_masks, y, x), dtype=np.uint8)
    print('Number of mask for training purposes: ' + str(num_of_masks))
    print('-' * 40)
    i = 0
    for m in sorted(os.listdir(mask_path)):
        mask = mpimg.imread(os.path.join(mask_path, m))
        # conversion to binary
        mask[mask < 1] = 0
        masks[i] = mask
        i += 1
        print('Prepared masks: {0} of {1} '.format(i, num_of_masks))
    print('-' * 40)
    np.save('masks.npy', masks)


def load_masks():
    return np.load('masks.npy')


if __name__ == '__main__':
    prepare_vectorized_data()
