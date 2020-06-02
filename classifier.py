import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

from scipy import ndimage
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.ensemble import BaggingClassifier
from features import load_features, load_features_info, load_masks
from ing_theme_matplotlib import mpl_style

TRAINING_TYPE = 'GRID_SEARCH'
NUM_OF_VALIDATION_IMAGES = 1
DEFAULT_C = 40
DEFAULT_GAMMA = 0.5
GRID_SEARCH_PARAMETERS = {'base_estimator__C': [1, 10, 25, 50, 100],
                          'base_estimator__gamma': [0.1, 1, 5, 10, 50]}


def choose_training_type():
    if TRAINING_TYPE == 'GRID_SEARCH':
        n_estimators = 16
        svc = BaggingClassifier(svm.SVC(max_iter=10000, tol=1, shrinking=0, cache_size=500, verbose=0, kernel='rbf'),
                                max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)
        return GridSearchCV(svc, GRID_SEARCH_PARAMETERS, verbose=2, n_jobs=-1, cv=2)

    elif TRAINING_TYPE == 'DEFAULT_TRAINING':
        n_estimators = 16
        return BaggingClassifier(
            svm.SVC(max_iter=25000, shrinking=0, cache_size=1000, verbose=0,
                    kernel='rbf', tol=0.5, gamma=DEFAULT_GAMMA, C=DEFAULT_C),
            max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)

    elif TRAINING_TYPE == 'FROM_SAVED_CLASSIFIER':
        return load(os.path.join('saved_models/', 'SVM_classifier.joblib'))

    else:
        print('No training has been selected!')
        return


def classify():
    print('-' * 30)
    print('TRAINING TYPE: {0}'.format(TRAINING_TYPE))
    print('-' * 30)
    # Load data and masks
    features = load_features()
    features_info = load_features_info()
    masks = load_masks()
    num_of_images = len(features_info)
    num_of_training_pixels = 0
    num_of_validation_pixels = 0

    assert (num_of_images > NUM_OF_VALIDATION_IMAGES)
    for i in range(num_of_images - NUM_OF_VALIDATION_IMAGES):
        num_of_training_pixels += features_info[i]['num_of_pixels']
    for i in range(num_of_images - NUM_OF_VALIDATION_IMAGES, num_of_images):
        num_of_validation_pixels += features_info[i]['num_of_pixels']
    print('Training data: {0} \nValidation data: {1} '.format(num_of_training_pixels, num_of_validation_pixels))

    # Standardize data
    x_train = features[:num_of_training_pixels]
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_train = preprocessing.normalize(x_train)
    x_validation = scaler.transform(features[num_of_training_pixels:])
    x_validation = preprocessing.normalize(x_validation)
    y_train = masks[:num_of_training_pixels]

    clf = choose_training_type()
    if TRAINING_TYPE != 'FROM_SAVED_CLASSIFIER':
        print('-' * 30)
        print('Training started...')
        start_time = time.time()
        clf.fit(x_train, y_train)
        print('-' * 30)
        print('Training ended: {:.2f} s'.format(time.time() - start_time))
        print('-' * 30)
        if TRAINING_TYPE == 'GRID_SEARCH':
            print("GRID SEARCH RESULTS\n")
            print('Best parameters: {}\n'.format(clf.best_params_))
            means = clf.cv_results_['mean_test_score']
            for mean, params in zip(means, clf.cv_results_['params']):
                print('Mean score: {:0.3f} Parameters: {}'.format(mean, params))
            print('-' * 30)

            scores = clf.cv_results_['mean_test_score'].reshape(len(GRID_SEARCH_PARAMETERS['base_estimator__C']),
                                                                len(GRID_SEARCH_PARAMETERS['base_estimator__gamma']))

            mpl_style(dark=True)
            # plt.figure(figsize=(10, 10))
            for ind, i in enumerate(GRID_SEARCH_PARAMETERS['base_estimator__C']):
                plt.plot(GRID_SEARCH_PARAMETERS['base_estimator__gamma'], scores[ind], label='C parameter: ' + str(i))
            plt.title('GRID SEARCH RESULTS')
            plt.xlabel('Gamma parameter')
            plt.ylabel('Mean score')
            plt.grid('on')
            plt.legend()
            plt.savefig('grid_search_results_figure.png', bbox_inches='tight', dpi=200)

        print('Saving model...')
        start_time = time.time()
        dump(clf, os.path.join('saved_models/', 'SVM_classifier.joblib'))
        print('Saving  ended: {:.2f} s'.format(time.time() - start_time))
        print('-' * 30)

    print('Predicting started...')
    print('-' * 30)
    start_time = time.time()
    predicted_masks = clf.predict(x_validation)
    print('Predicting ended: {:.2f} s'.format(time.time() - start_time))
    print('-' * 30)
    previous_mask_pixels = 0
    current_num_of_pixels = 0
    masks_predicted = 0
    # Saving predicted and truth masks as pairs
    # There is a need for converting predicted vector to 2D masks
    for i in range(num_of_images - NUM_OF_VALIDATION_IMAGES, num_of_images):
        current_num_of_pixels = features_info[i]['num_of_pixels']
        predicted_mask = np.asarray(predicted_masks)[previous_mask_pixels:previous_mask_pixels + current_num_of_pixels]
        predicted_mask = predicted_mask.reshape(features_info[i]['height'], features_info[i]['width'])
        predicted_mask = ndimage.binary_opening(predicted_mask)
        predicted_mask = ndimage.binary_closing(predicted_mask)
        masks_predicted += 1
        plt.imsave(os.path.join('predictions/', 'predicted_mask_' + str(masks_predicted) + '.png'), predicted_mask,
                   cmap='gray')

        offset = num_of_training_pixels + previous_mask_pixels
        truth_mask = masks[offset:offset + current_num_of_pixels].reshape(
            features_info[i]['height'], features_info[i]['width'])
        plt.imsave(os.path.join('truth/', 'truth_mask_' + str(masks_predicted) + '.png'), truth_mask, cmap='gray')
        previous_mask_pixels = current_num_of_pixels


def dice(ref, pred):
    intersection = np.logical_and(ref, pred)
    return 2. * intersection.sum() / (ref.sum() + pred.sum())


def classifier_score():
    sum_dice = 0
    for i in range(NUM_OF_VALIDATION_IMAGES):
        prediction = plt.imread(os.path.join('predictions/', 'predicted_mask_' + str(i + 1) + '.png'))
        truth = plt.imread(os.path.join('truth/', 'truth_mask_' + str(i + 1) + '.png'))
        sum_dice += dice(truth, prediction)
    print('Dice coefficient {:.3f}'.format(sum_dice / NUM_OF_VALIDATION_IMAGES))
    print('-' * 30)


if __name__ == '__main__':
    PATH = sys.argv[1]
    os.chdir(PATH)
    classify()
    classifier_score()
