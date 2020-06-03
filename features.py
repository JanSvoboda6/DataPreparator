import math
import os
import sys
import matplotlib.image
import numpy as np
import pydicom
from scipy import ndimage

MATRIX_SIZE = 7


def to_signed(image):
    minimum = np.min(image)
    if minimum < 0:
        image = image + abs(minimum)
    return image


def normalize(image):
    # This method normalizes value of each pixel, converts image to unsigned int
    # to get only 256 possible pixel values and then normalizes values to range between 0-1
    image = image.astype(np.float64)
    image /= np.max(image)
    image *= 255

    image = image.astype(np.uint8)

    image = image.astype(np.float32)
    image /= 255
    return image


def equalize_histogram(image):
    reshaped_image = image.ravel()
    histogram, bins = np.histogram(reshaped_image, bins=10, density=True)
    cum_sum = histogram.cumsum()
    # Normalize cumulative sum, value at last index is maximum
    cum_sum /= cum_sum[-1]
    equalized_image = np.interp(reshaped_image, bins[:-1], cum_sum)
    equalized_image = equalized_image / np.max(equalized_image)
    unique, counts = np.unique(equalized_image, return_counts=True)
    return equalized_image.reshape(image.shape)


def count_gradient(image):
    assert (MATRIX_SIZE % 2 == 1)
    gradients_x = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    gradients_y = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
    gradients_feature = np.zeros((image.size, MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    middle = math.trunc(MATRIX_SIZE / 2)

    for i in range(image.shape[0]):
        for j in range(image.shape[1] - 1):
            gradients_x[i, j + 1] = math.fabs((image[i, j + 1]) - (image[i, j]))

    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1]):
            gradients_y[i + 1, j] = math.fabs((image[i + 1, j]) - (image[i, j]))

    gradients_x = np.pad(gradients_x, middle, mode='constant')
    gradients_y = np.pad(gradients_y, middle, mode='constant')
    gradients_sum = (np.add(gradients_x, gradients_y)) / 2
    pixel = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            gradients_feature[pixel] = gradients_sum[i:i + MATRIX_SIZE, j:j + MATRIX_SIZE]
            pixel += 1
    return gradients_feature.reshape(-1, MATRIX_SIZE * MATRIX_SIZE)


def gaussian_filtering(image):
    gaussian = ndimage.gaussian_filter(image, sigma=2)
    gaussian /= np.max(gaussian)
    return gaussian.ravel()


def median_filtering(image):
    median = ndimage.median_filter(image, size=7)
    median /= np.max(median)
    return median.ravel()


def sobel_operator(image):
    sobel = ndimage.sobel(image)
    sobel /= np.max(sobel)
    return sobel.ravel()


def mean(image):
    mean_value = ndimage.uniform_filter(image, (MATRIX_SIZE, MATRIX_SIZE))
    mean_value /= np.max(mean_value)
    return mean_value.ravel()


def variance(image):
    # Implementation of and alternative variance formula: var = sum(X^2)/N - μ^2
    mean_values = ndimage.uniform_filter(image, (MATRIX_SIZE, MATRIX_SIZE))
    sqr_mean = ndimage.uniform_filter(image ** 2, (MATRIX_SIZE, MATRIX_SIZE))
    var = sqr_mean - mean_values ** 2
    var /= np.max(var)
    return var.ravel()


def laplacian(image):
    laplacian_values = ndimage.laplace(image)
    laplacian_values /= np.max(laplacian_values)
    return laplacian_values.ravel()


def getFeatureDictionary(image):
    return {
        'voxelValue': np.array(image).ravel(),
        'mean': mean(image),
        'variance': variance(image),
        'gaussianFilter': gaussian_filtering(image),
        'medianFilter': median_filtering(image),
        'sobelOperator': sobel_operator(image),
        'gradientMatrix': count_gradient(image),
        'laplacian': laplacian(image)
    }


def prepare_features():
    data_path = os.path.abspath('train_data/')
    num_of_images = len(os.listdir(data_path))
    features = []
    features_info = []
    images_prepared = 0

    print('Preparing feature vector...')
    print('-' * 30)
    for image_path in sorted(os.listdir(data_path)):
        image = pydicom.dcmread(os.path.join(data_path, image_path)).pixel_array
        image = to_signed(image)
        image = normalize(image)
        image = equalize_histogram(image)
        featureDictionary = getFeatureDictionary(image)
        # For each pixel
        for i in range(image.size):
            feature_row = []
            for feature in featureDictionary.values():
                if isinstance(feature[0], np.ndarray):
                    feature_row.extend(feature[i])
                else:
                    feature_row.append(feature[i])
            features.append(feature_row)
        # Append image information also to features_info, which is used to store and retrieve
        # necessary information about images during classifying
        info = {
            'num_of_pixels': image.size,
            'height': image.shape[0],
            'width': image.shape[1],
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
    for mask in sorted(os.listdir(mask_path)):
        mask = matplotlib.image.imread(os.path.join(mask_path, mask))
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        assert (mask.ndim == 2)
        # Conversion to binary
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
    # Change path to each modality
    PATH = sys.argv[1]
    os.chdir(PATH)
    prepare_features()
    prepare_mask()
