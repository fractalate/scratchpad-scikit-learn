import pandas as pd
import os
import pickle

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

SOURCE_FILE = 'Wine_Quality_Data.csv'

TRAINING_FILE = './results/training.pickle'
TEST_FILE = './results/validation.pickle'
MODEL_KNN_FILE = './results/model_knn.pickle'

def make_results_dir():
    if not os.path.isdir('./results'):
        os.mkdir('./results')

def split_Xy(data_set):
    X = data_set.drop('color', axis=1)
    y = data_set['color']
    return X, y

def split_dataset():
    make_results_dir()
    if os.path.isfile(TRAINING_FILE) and os.path.isfile(TEST_FILE):
        print('skipping split_dataset()')
        return

    if not os.path.isfile(SOURCE_FILE):
        raise Exception(SOURCE_FILE + ' is missing or is not a file. See README.md for where to get this file.')
    df = pd.read_csv(SOURCE_FILE)

    # 794 is just a random number chosen by me.
    training_set, test_set = train_test_split(df, test_size=0.2, random_state=794)
    with open(TRAINING_FILE, 'wb') as fout:
        pickle.dump(training_set, fout)
    with open(TEST_FILE, 'wb') as fout:
        pickle.dump(test_set, fout)

def train_model_knn():
    if os.path.isfile(MODEL_KNN_FILE):
        print('skipping train_model_knn()')
        return

    with open(TRAINING_FILE, 'rb') as fin:
        training_set = pickle.load(fin)
    X, y = split_Xy(training_set)

    clf = KNeighborsClassifier()
    clf.fit(X, y)

    with open(MODEL_KNN_FILE, 'wb') as fout:
        pickle.dump(clf, fout)

def assess_model_knn():
    with open(MODEL_KNN_FILE, 'rb') as fin:
        clf = pickle.load(fin)

    with open(TRAINING_FILE, 'rb') as fin:
        test_set = pickle.load(fin)
    X, y = split_Xy(test_set)

    y_pred = clf.predict(X)

    accuracy = accuracy_score(y, y_pred)
    print('Accuracy: {:.2f}%'.format(accuracy * 100))

    report = classification_report(y, y_pred)
    print('Classification Report:')
    print(report)

    conf_matrix = confusion_matrix(y, y_pred)
    print('Confusion Matrix:')
    print(conf_matrix)

def main():
    split_dataset()
    train_model_knn()
    assess_model_knn()

if __name__ == '__main__':
    main()
