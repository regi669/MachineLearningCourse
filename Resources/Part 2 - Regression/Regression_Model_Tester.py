import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

dataset_name = 'Data.csv'


def prepare_dataset(dataset_name):
    dataset = pd.read_csv(dataset_name)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    return X_train, X_test, y_train, y_test


def linear_regression(X_train, X_test, y_train, y_test):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return r2_score(y_test, y_pred)


def polynomial_regression(X_train, X_test, y_train, y_test):
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X_train)
    regressor = LinearRegression()
    regressor.fit(X_poly, y_train)
    y_pred = regressor.predict(poly_reg.transform(X_test))
    return r2_score(y_test, y_pred)


def support_vector_regression(X_train, X_test, y_train, y_test):
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    y_train = y_train.reshape(len(y_train), 1)
    y_test = y_test.reshape(len(y_test), 1)
    X_train = sc_X.fit_transform(X_train)
    y_train = sc_y.fit_transform(y_train)
    regressor = SVR(kernel='rbf')
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(sc_X.transform(X_test))
    y_pred = sc_y.inverse_transform([y_pred])
    y_pred = y_pred.reshape((len(y_test), 1))
    return r2_score(y_test, y_pred)


def decision_tree_regresion(X_train, X_test, y_train, y_test):
    regressor = DecisionTreeRegressor()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return r2_score(y_test, y_pred)


def random_forest_regression(X_train, X_test, y_train, y_test):
    regressor = RandomForestRegressor(n_estimators=100)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    return r2_score(y_test, y_pred)


X_train, X_test, y_train, y_test = prepare_dataset(dataset_name)
print('Linear Regression R2 Score: {}'.format(linear_regression(X_train, X_test, y_train, y_test)))
print('Polynomial Regression R2 Score: {}'.format(polynomial_regression(X_train, X_test, y_train, y_test)))
print('Support Vector Regression R2 Score: {}'.format(support_vector_regression(X_train, X_test, y_train, y_test)))
print('Decision Regression R2 Score: {}'.format(decision_tree_regresion(X_train, X_test, y_train, y_test)))
print('Random Forest Regression R2 Score: {}'.format(random_forest_regression(X_train, X_test, y_train, y_test)))
