import glob
from sklearn.model_selection import train_test_split
import shutil
import os
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import numpy as np

dir_path = r"C:\IDC_regular_ps50_idx5"
train_test_dir = "../train_test"
train_test_dir_balanced = "../train_test_balanced"
validate_dir = "../validate"


def initial_split(class_label=1):
    paths = glob.glob(f'{dir_path}/**/{class_label}/*.png')
    labels = [int(path[-5]) for path in paths]
    print(sum(labels))

    X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=.1, random_state=42)

    for x in X_val:
        shutil.copy2(x, validate_dir)
    print("validation files copied!")

    for x in X_train:
        shutil.copy2(x, train_test_dir)
    print("train_test files copied!")


def prepare_dirs():
    if not os.path.exists(train_test_dir):
        os.makedirs(train_test_dir)
    if not os.path.exists(validate_dir):
        os.makedirs(validate_dir)
    if not os.path.exists(train_test_dir_balanced):
        os.makedirs(train_test_dir_balanced)


def split_dataset():
    prepare_dirs()
    initial_split(0)
    initial_split(1)


def random_under_sampling():
    X = glob.glob(f'{train_test_dir}/*.png')
    y = [int(path[-5]) for path in X]
    X = np.array(X).reshape(-1, 1)
    print('Original dataset shape %s' % Counter(y))

    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    print('Resampled dataset shape %s' % Counter(y_resampled))
    X_resampled = X_resampled.flatten().tolist()
    for x in X_resampled:
        shutil.copy2(x, train_test_dir_balanced)
    print("balanced files copied!")


def balance_dataset():
    prepare_dirs()
    random_under_sampling()


def prepare_data():
    split_dataset()
    balance_dataset()


prepare_data()
