import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import RandomizedSearchCV

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


def get_experiment_data(train_features, test_features, train_idx, test_idx):
    X_train = np.zeros(shape=(len(train_idx), train_features.shape[1] - 3))
    print('train shape', X_train.shape)
    unique_train_idx = np.unique(train_idx)
    y_train = np.zeros(shape=len(train_idx))
    for i, pid in enumerate(unique_train_idx):
        k = np.argwhere(train_features[:, 2] == pid)
        for j in k:
            X_train[j[0], :] = np.squeeze(train_features[j[0], 3:])
            y_train[j[0]] = (0 if int(train_features[j[0], 1]) < 24 else 1)

    X_test = np.zeros(shape=(len(test_idx), test_features.shape[1] - 3))
    unique_test_idx = np.unique(test_idx)
    y_test = np.zeros(shape=len(test_idx))
    for i, pid in enumerate(unique_test_idx):
        k = np.argwhere(test_features[:, 2] == pid)
        for j in k:
            X_test[j[0], :] = np.squeeze(test_features[j[0], 3:])
            y_test[j[0]] = (0 if int(test_features[j[0], 1]) < 24 else 1)

    print('test shape', X_test.shape)

    return X_train, y_train, X_test, y_test

def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
    return results['mean_test_score'][np.flatnonzero(results['rank_test_score'] == 1)[0]]

def optimize(X, y, clf, param_dist, name, n_iter_search=20):
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist,
                                       scoring='accuracy', n_iter=n_iter_search, cv=5)
    random_search.fit(X, y)
    best_sol = report(random_search.cv_results_)
    print(name, best_sol)

def make_experiment(train_features, test_features, train_idx, test_idx):
    X_train, y_train, X_test, y_test = get_experiment_data(train_features, test_features, train_idx, test_idx)

    # KNN
    clf = KNeighborsClassifier()
    param_dist = {"n_neighbors": np.arange(1, 21, 1)}
    optimize(X_train, y_train, clf, param_dist, 'KNN')

    # Random forest
    clf = RandomForestClassifier()
    param_dist = {"n_estimators": np.arange(1, 50, 5),
                  "max_depth": np.arange(1, 3, 1)}
    optimize(X_train, y_train, clf, param_dist, 'RF')

    # AdaBoost
    clf = AdaBoostClassifier()
    param_dist = {"n_estimators": np.arange(1, 200, 5),
                  "learning_rate": np.arange(0.1, 1.1, 0.2)}
    optimize(X_train, y_train, clf, param_dist, 'Ada', 50)

    # SVM
    clf = SVC()
    param_dist = {"kernel": ['linear', 'poly', 'rbf'],
                  "C": [10 ** (_) for _ in np.arange(-5, 6, 1, dtype=float)],
                  "degree": np.arange(1, 7, 1)}
    optimize(X_train, y_train, clf, param_dist, 'SVM', 30)


FEATURES_DIR = 'F:/src/features'
ROOT_DIR = 'F:/' #TODO: change
data_path = os.path.join(ROOT_DIR, 'kt_dataset_dicom')

train_files_list = ROOT_DIR + 'dicom_train_list.txt'
val_files_list = ROOT_DIR + 'val_list_id.txt'
half_val_files_list = ROOT_DIR + 'half_sized_val_list.txt'

aec2d_features_fp = os.path.join(FEATURES_DIR, 'f_aec2d_deeper_half_size_256.txt')
aec2d_test_features_fp = os.path.join(FEATURES_DIR, 'f_aec2d_deeper_half_size_256_test.txt')

aec2d_features = np.loadtxt(aec2d_features_fp, delimiter=',', dtype=str)
aec2d_test_features = np.loadtxt(aec2d_test_features_fp, delimiter=',', dtype=str)
print('features shape: ', aec2d_features.shape)
print(aec2d_test_features.shape)
aec2d_ids = set(aec2d_features[:, 2])
aec2d_test_ids = set(aec2d_test_features[:, 2])

train_file = open(train_files_list, 'r')
val_file = open(half_val_files_list, 'r')
train_idx = []
test_idx = []

for line in train_file.readlines():
    line = line.split('_')[1]
    train_idx.append(line.split('/')[0])

for line in val_file.readlines():
    line = line.split('_')[1]
    test_idx.append(line.split('/')[0])

print("\nWORKING ON AEC FEATURES\n")
make_experiment(aec2d_features, aec2d_test_features, train_idx, test_idx)
