import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import jaccard_score
from sklearn import svm
from data_preparator import load_train_data, load_masks


def train_classifier():
    x = load_train_data()
    y = load_masks()
    images = []
    masks = []
    for i in range(0, x.shape[0]):
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
    y_prediction = clf.predict(images)
    # acc = metrics.accuracy_score(y[0].ravel().tolist(), y_prediction)
    np.save('PREDICTED MASK.npy', y_prediction)
    print('-' * 20)


def dice(truth, predicted):
    intersection = np.logical_and(truth, predicted)
    return 2. * intersection.sum() / (truth.sum() + predicted.sum())


def jaccard(truth, predicted):
    return jaccard_score(truth, predicted)


if __name__ == '__main__':
    train_classifier()
    mask = np.load('PREDICTED MASK.npy')
    truth = np.load('masks.npy').ravel()
    print('DICE coeficient')
    print(dice(truth, mask))
    print('-' * 20)
    print('JACCARD coeficient')
    print(jaccard(truth, mask))
    print('-' * 20)

    mask = mask.reshape(256, 315)
    plt.imshow(mask)
    plt.show(block=False)
    plt.pause(2)
    plt.close('all')
