# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:28:33 2023

@author: brmor
"""

import pandas as pd
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier


class titanic_preds():
    def __init__(self):
        self.csv_train_dtypes = {'passenger_id': int,
                                 'survived': int,
                                 'p_class': int,
                                 'name': str,
                                 'sex': str,
                                 'age': float,
                                 'sib_sp': int,
                                 'par_ch': int,
                                 'ticket': str,
                                 'fare': float,
                                 'cabin': str,
                                 'embarked': str}
        self.csv_test_dtypes = {'passenger_id': int,
                                'p_class': int,
                                'name': str,
                                'sex': str,
                                'age': float,
                                'sib_sp': int,
                                'par_ch': int,
                                'ticket': str,
                                'fare': float,
                                'cabin': str,
                                'embarked': str}
        self.X_train = pd.DataFrame()
        self.y_train = pd.Series()
        self.X_test = pd.DataFrame()
        self.y_preds = pd.Series()

    def set_x_train(self, x):
        self.X_train = x

    def get_x_train(self):
        return self.X_train

    def set_y_train(self, x):
        self.y_train = x

    def get_y_train(self):
        return self.y_train

    def set_x_test(self, x):
        self.X_test = x

    def get_x_test(self):
        return self.X_test

    def set_preds(self, x):
        self.y_preds = x

    def get_preds(self):
        return self.y_preds

    def run_naive_bayes(self, bln_fit, bln_predict):
        if bln_fit:
            self.clf_nb = MultinomialNB()
            self.clf_nb.fit(self.X_train, self.y_train)
        if bln_predict:
            self.y_preds = self.clf_nb.predict(self.X_test)

    def run_random_forest(self, bln_fit, bln_predict):
        if bln_fit:
            self.clf_rf = RandomForestClassifier()
            self.clf_rf.fit(self.X_train, self.y_train)
        if bln_predict:
            self.y_preds = self.clf_rf.predict(self.X_test)

    def run_logistic_regression(self, bln_fit, bln_predict):
        if bln_fit:
            self.logr = linear_model.LogisticRegression()
            self.logr.fit(self.X_train, self.y_train)
        if bln_predict:
            self.y_preds = self.logr.predict(self.X_test)
