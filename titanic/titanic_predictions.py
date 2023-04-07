# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 21:28:33 2023

@author: brmor
"""

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier


class titanic_preds():
    def __init__(self, run_grid_search=False, param_grid=None):
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
        self.bln_run_grid_search = run_grid_search
        self.param_grid = param_grid
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

    def check_corr_features(self, df, cut_val):
        df_corr = df.corr()
        df_corr_filter = np.ones([len(df_corr)]*2, dtype=bool)
        df_corr_filter = np.triu(df_corr_filter)
        df_corr_filter = (df_corr.mask(df_corr_filter).abs() > cut_val).any()
        df_corr_cols = pd.DataFrame(df_corr_filter,
                                    columns=['corr']
                                    ).reset_index().rename(
                                            columns={'index': 'corr_col'})
        obj_drop_cols = df_corr_cols[df_corr_cols['corr']
                                     ]['corr_col'].to_list()
        return obj_drop_cols

    def run_naive_bayes(self, bln_fit, bln_predict):
        if bln_fit:
            self.clf_nb = MultinomialNB()
            self.clf_nb.fit(self.X_train, self.y_train)
        if bln_predict:
            self.y_preds = self.clf_nb.predict(self.X_test)

    def run_random_forest(self, bln_fit, bln_predict):
        if self.bln_run_grid_search:
            if self.param_grid is None:
                self.param_grid = {'n_estimators': [25, 50, 100, 150],
                                   'max_features': ['sqrt', 'log2', None],
                                   'max_depth': [3, 6, 9],
                                   'max_leaf_nodes': [3, 6, 9]
                                   }
            grid_search = GridSearchCV(RandomForestClassifier(),
                                       param_grid=self.param_grid)
            grid_search.fit(self.X_train, self.y_train)
            self.clf_rf = grid_search.best_estimator_
            bln_fit = False
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
