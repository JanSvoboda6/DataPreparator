import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os
from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
from joblib import dump, load
from sklearn.ensemble import BaggingClassifier
from joblib.parallel import cpu_count, Parallel, delayed
from features import load_features, load_features_info, load_masks

matplotlib.use('Agg')

TRAINING_TYPE = 'DEFAULT_CLASSIFIER'
NUM_OF_TESTING_IMAGES = 1
DEFAULT_GAMMA = 0.5
DEFAULT_C = 40
GRID_SEARCH_PARAMETERS = {'base_estimator__gamma': [0.01, 0.1, 0.5, 1, 10],
                          'base_estimator__C': [0.01, 0.1, 1, 10, 30, 50]}


def choose_training_type():
    if TRAINING_TYPE == 'GRID_SEARCH':
        n_estimators = 16
        svc = BaggingClassifier(svm.SVC(max_iter=15000, tol=0.5, shrinking=0, cache_size=1000, verbose=0, kernel='rbf'),
                                max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)
        return GridSearchCV(svc, GRID_SEARCH_PARAMETERS, verbose=3, n_jobs=-1, cv=2)

    elif TRAINING_TYPE == 'DEFAULT_TRAINING':
        n_estimators = 16
        return BaggingClassifier(
            svm.SVC(max_iter=25000, shrinking=0, cache_size=500, verbose=3,
                    kernel='rbf', tol=0.5, gamma=DEFAULT_GAMMA, C=DEFAULT_C),
            max_samples=1.0 / n_estimators, n_estimators=n_estimators, n_jobs=-1)

    elif TRAINING_TYPE == 'FROM_SAVED_CLASSIFIER':
        return load(os.path.join('saved_models/', 'SVM_classifier.joblib'))

    else:
        print('No training has been selected!')
        return


def classify():
    print('-' * 20)
    print('TRAINING TYPE: {0}'.format(TRAINING_TYPE))
    print('-' * 20)
    # Load data and masks
    features = load_features()
    features_info = load_features_info()
    masks = load_masks()
    num_of_images = len(features_info)
    num_of_training_pixels = 0
    num_of_testing_pixels = 0
    assert (num_of_images > NUM_OF_TESTING_IMAGES)
    for i in range(num_of_images - NUM_OF_TESTING_IMAGES):
        num_of_training_pixels += features_info[i]['num_of_pixels']
    for i in range(num_of_images - NUM_OF_TESTING_IMAGES, num_of_images):
        num_of_testing_pixels += features_info[i]['num_of_pixels']
    print('Training data: {0} \nTesting data: {1} '.format(num_of_training_pixels, num_of_testing_pixels))
    # Standardize data
    x_train = features[:num_of_training_pixels]
    scaler = preprocessing.StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_train = preprocessing.normalize(x_train)
    x_testing = scaler.transform(features[num_of_training_pixels:])
    x_testing = preprocessing.normalize(x_testing)
    y_train = masks[:num_of_training_pixels]

    clf = choose_training_type()

    if TRAINING_TYPE != 'FROM_SAVED_CLASSIFIER':
        print('-' * 20)
        print('Training started')
        print('-' * 20)
        start_time = time.time()
        clf.fit(x_train, y_train)
        print('Training ended: %s' % (time.time() - start_time))
        print('-' * 20)
        if TRAINING_TYPE == 'GRID_SEARCH':
            print('Best parameters')
            print(clf.best_params_)
            print('-' * 20)
        print('Saving model...')
        start_time = time.time()
        dump(clf, os.path.join('saved_models/', 'SVM_classifier.joblib'))
        print('Saving  ended: %s' % (time.time() - start_time))
        print('-' * 20)
    print('Predicting started')
    print('-' * 20)
    start_time = time.time()
    predicted_masks = clf.predict(x_testing)
    # predicted_masks = parallel_predict(clf, x_testing, -1)
    print('Predicting ended: %s' % (time.time() - start_time))
    previous_mask_pixels = 0
    current_num_of_pixels = 0
    masks_predicted = 0
    for i in range(num_of_images - NUM_OF_TESTING_IMAGES, num_of_images):
        current_num_of_pixels = features_info[i]['num_of_pixels']
        predicted_mask = np.asarray(predicted_masks)[previous_mask_pixels:previous_mask_pixels + current_num_of_pixels]
        predicted_mask = predicted_mask.reshape(features_info[i]['height'], features_info[i]['width'])
        masks_predicted += 1
        np.save(os.path.join('predictions/', 'predicted_mask_' + str(masks_predicted) + '.npy'), predicted_mask)
        offset = num_of_training_pixels + previous_mask_pixels
        truth_mask = masks[offset:offset + current_num_of_pixels].reshape(
            features_info[i]['height'], features_info[i]['width'])
        np.save(os.path.join('truth/', 'truth_mask_' + str(++masks_predicted) + '.npy'), truth_mask)
        previous_mask_pixels = current_num_of_pixels
    print('-' * 20)


def parallel_predict(classifier, x_testing, n_jobs=1):
    num_of_jobs = max(cpu_count() + 1 + n_jobs, 1)
    batch = int(np.ceil(len(x_testing) / num_of_jobs))
    print('len of testing ' + str(len(x_testing)))
    print('batch size ' + str(batch))
    print('num_of_jobs ' + str(num_of_jobs))
    x_testing = np.array_split(x_testing, 10)
    parallel = Parallel(n_jobs=num_of_jobs)
    jobs = (delayed(classifier.predict)(x_testing[i])
            for i in range(len(x_testing)))
    results = parallel(jobs)
    return np.concatenate(results)


def dice(ref, pred):
    intersection = np.logical_and(ref, pred)
    return 2. * intersection.sum() / (ref.sum() + pred.sum())


def classifier_score():
    info = load_features_info()
    num_of_images = len(info)
    sum_dice = 0
    for i in range(NUM_OF_TESTING_IMAGES):
        prediction = np.load(os.path.join('predictions/', 'predicted_mask_' + str(i + 1) + '.npy')).ravel()
        prediction = prediction.reshape(info[num_of_images - NUM_OF_TESTING_IMAGES + i]['height'],
                                        info[num_of_images - NUM_OF_TESTING_IMAGES + i]['width'])
        plt.imshow(prediction)
        plt.title('Predicted #' + str(i + 1))
        plt.savefig(os.path.join('predictions/', 'img_predicted_mask_' + str(i + 1) + '.png'))

        truth = np.load(os.path.join('truth/', 'truth_mask_' + str(i + 1) + '.npy')).ravel()
        truth = truth.reshape(info[num_of_images - NUM_OF_TESTING_IMAGES + i]['height'],
                              info[num_of_images - NUM_OF_TESTING_IMAGES + i]['width'])
        plt.imshow(truth)
        plt.title('Truth #' + str(i + 1))
        plt.savefig(os.path.join('truth/', 'img_truth_mask_' + str(i + 1) + '.png'))
        sum_dice += dice(truth, prediction)
    print('Dice coefficient')
    print(sum_dice / NUM_OF_TESTING_IMAGES)
    print('-' * 20)


if __name__ == '__main__':
    PATH = sys.argv[1]
    os.chdir(PATH)
    classify()
    classifier_score()
