import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

dataset_name = 'Data.csv'


def prepare_dataset(dataset_name):
    dataset = pd.read_csv(dataset_name)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    return X_train, X_test, y_train, y_test


def logistic_regression(X_train, X_test, y_train, y_test):
    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    return cm, acc_score


def k_nearest_neighbors(X_train, X_test, y_train, y_test):
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    return cm, acc_score


def support_vector_machine_linear(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel='linear')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    return cm, acc_score


def support_vector_machine_rbf(X_train, X_test, y_train, y_test):
    classifier = SVC(kernel='rbf')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    return cm, acc_score


def naive_bayes(X_train, X_test, y_train, y_test):
    classifier = GaussianNB()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    return cm, acc_score


def decision_tree(X_train, X_test, y_train, y_test):
    classifier = DecisionTreeClassifier(criterion='entropy')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    return cm, acc_score


def random_forest(X_train, X_test, y_train, y_test):
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy')
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc_score = accuracy_score(y_test, y_pred)
    return cm, acc_score


X_train, X_test, y_train, y_test = prepare_dataset(dataset_name)


print('Logistic Regression Classification Score: {}'.format(logistic_regression(X_train, X_test, y_train, y_test)))
print('K Nearest Neighbors Classification Score: {}'.format(k_nearest_neighbors(X_train, X_test, y_train, y_test)))
print('SVM Linear Classification Score: {}'.format(support_vector_machine_linear(X_train, X_test, y_train, y_test)))
print('SVM RBF Classification Score: {}'.format(support_vector_machine_rbf(X_train, X_test, y_train, y_test)))
print('Naive Bayes Classification Score: {}'.format(naive_bayes(X_train, X_test, y_train, y_test)))
print('Decision Tree Classification Score: {}'.format(decision_tree(X_train, X_test, y_train, y_test)))
print('Random Forest Classification Score: {}'.format(random_forest(X_train, X_test, y_train, y_test)))




