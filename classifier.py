import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import jaccard_score
from data_preparator import load_train_data, load_masks


def classify():
    x = load_train_data()
    y = load_masks()
    images = []
    masks = []
    # conversion of 3D array to 1D list
    for i in range(0, x.shape[0] - 1):
        images.extend(x[i].reshape(-1, 1).tolist())
        masks.extend(y[i].ravel().tolist())

    print('-' * 20)
    print('Training started')
    print('-' * 20)
    clf = svm.SVC(gamma='scale', shrinking=0, verbose=1)
    clf.fit(images, masks)
    print('-' * 20)
    print('Predicting started')
    print('-' * 20)
    last_image = x[-1].reshape(-1, 1).tolist()
    prediction = clf.predict(last_image)
    # conversion from 1D list to 2D array
    np.asarray(prediction).reshape(256, 315)
    np.save('predicted_mask.npy', prediction)
    print('-' * 20)


def dice(reference, prediction):
    intersection = np.logical_and(reference, prediction)
    return 2. * intersection.sum() / (reference.sum() + prediction.sum())


def jaccard(reference, prediction):
    return jaccard_score(reference, prediction)


if __name__ == '__main__':
    #classify()
    predicted_mask = np.load('predicted_mask.npy').ravel()
    truth_mask = np.load('masks.npy')[-1].ravel()
    print('Dice coefficient')
    print(dice(truth_mask, predicted_mask))
    print('-' * 20)
    print('Jaccard coefficient')
    print(jaccard(truth_mask, predicted_mask))
    print('-' * 20)

    predicted_mask = predicted_mask.reshape(256, 315)
    plt.figure(1)
    plt.title('PREDICTED')
    plt.imshow(predicted_mask)

    truth_mask = truth_mask.reshape(256, 315)
    plt.figure(2)
    plt.title('REFERENCE')
    plt.imshow(truth_mask)

    plt.show()
    plt.close('all')
