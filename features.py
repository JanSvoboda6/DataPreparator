import math
import os
import sys
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import pickle
from scipy.ndimage import gaussian_filter


def to_signed(img):
    minimum = np.min(img)
    if minimum < 0:
        img = img + abs(minimum)
    return img


def normalize(img):
    img = img.astype(np.float32)
    img /= 255
    return img


def count_gradient(img, matrix_size):
    assert (matrix_size % 2 == 1)
    gradients_x = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gradients_y = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)
    gradients_feature = np.zeros((img.size, matrix_size, matrix_size), dtype=np.float32)
    middle = math.trunc(matrix_size / 2)

    for i in range(img.shape[0]):
        for j in range(img.shape[1] - 1):
            gradients_x[i, j + 1] = math.fabs((img[i, j + 1]) - (img[i, j]))

    for i in range(img.shape[0] - 1):
        for j in range(img.shape[1]):
            gradients_y[i + 1, j] = math.fabs((img[i + 1, j]) - (img[i, j]))

    gradients_x = np.pad(gradients_x, middle, mode='constant')
    gradients_y = np.pad(gradients_y, middle, mode='constant')
    gradients_sum = (np.add(gradients_x, gradients_y)) / 2
    pixel = 0
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gradients_feature[pixel] = gradients_sum[i:i + matrix_size, j:j + matrix_size]
            pixel += 1
    return gradients_feature.reshape(-1, matrix_size * matrix_size)


def convolution(img):
    gaussian = gaussian_filter(img, sigma=2)
    gaussian /= np.max(gaussian)
    return gaussian.ravel()


def prepare_features():
    data_path = os.path.abspath('train_data/')
    num_of_images = len(os.listdir(data_path))
    features = []
    features_info = []
    i = 0
    images_prepared = 0
    print('Preparing feature vector...')
    print('-' * 30)
    for image in sorted(os.listdir(data_path)):
        data = pydicom.dcmread(os.path.join(data_path, image)).pixel_array
        data = to_signed(data)
        data = normalize(data)
        gaussian = convolution(data)
        gradients = count_gradient(data, 7)
        for j in range(data.size):
            features.append([gaussian[j]])
            # features[i].extend(data.reshape(data.size, -1)[j])
            features[i].extend(gradients[j])
            i += 1
            j += 1

        info = {
            'num_of_pixels': data.size,
            'height': data.shape[0],
            'width': data.shape[1]
        }
        features_info.append(info)
        images_prepared += 1
        print('Prepared images: {0} of {1} '.format(images_prepared, num_of_images))
    print('-' * 30)
    f = open('features_info.txt', 'w')
    f.write(str(features_info))
    f.close()
    features = np.asarray(features)
    np.save('features.npy', features)
    print('Feature vector has been saved.')


def prepare_mask():
    mask_path = os.path.abspath('masks/')
    num_of_masks = len(os.listdir(mask_path))
    masks = []
    print('Preparing masks vector...')
    print('-' * 30)
    i = 0
    for m in sorted(os.listdir(mask_path)):
        mask = matplotlib.image.imread(os.path.join(mask_path, m))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        assert (mask.ndim == 2)
        # conversion to binary
        mask[mask < 1] = 0
        mask = mask.astype(np.uint8)
        mask = mask.ravel()
        masks.extend(mask)
        i += 1
        print('Prepared masks: {0} of {1} '.format(i, num_of_masks))
    print('-' * 30)
    masks = np.asarray(masks)
    np.save('masks.npy', masks)
    print('Vector of masks has been saved.')


def load_features():
    return np.load('features.npy')


def load_features_info():
    f = open('features_info.txt', 'r')
    data = f.read()
    f.close()
    return eval(data)


def load_masks():
    return np.load('masks.npy')


if __name__ == '__main__':
    PATH = sys.argv[1]
    os.chdir(PATH)
    prepare_features()
    prepare_mask()
