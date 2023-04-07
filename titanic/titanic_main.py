# -*- coding: utf-8 -*-
"""
Created on Sat Apr  1 20:40:16 2023

@author: brmor

read in titanic data and predict if passengers will survive
"""

import json
import logging
import argparse
import pandas as pd
import titanic_predictions as ttp

from sklearn import metrics
from sklearn.model_selection import train_test_split


def run_titanic_preds(args):
    logging.info('Define Titanic Predictions Object')
    obj_preds = ttp.titanic_preds(args.bln_run_grid_search, args.param_grid)

    logging.info('Reading in training and test data')
    df_train_data = pd.read_csv(args.str_training_data,
                                names=list(obj_preds.csv_train_dtypes.keys()),
                                header=0,
                                dtype=obj_preds.csv_train_dtypes)
    y_train = df_train_data['survived']
    df_train_data = df_train_data.drop(columns={'survived'})
    df_test_data = pd.read_csv(args.str_test_data,
                               names=list(obj_preds.csv_test_dtypes.keys()),
                               header=0,
                               dtype=obj_preds.csv_test_dtypes)

    logging.info('Clean features')
    df_train_data['name'] = df_train_data['name'].apply(
            lambda x: x.split(', ')[0].lower())
    df_train_data = df_train_data.fillna(0)
    df_test_data['name'] = df_test_data['name'].apply(
            lambda x: x.split(', ')[0].lower())
    df_test_data = df_test_data.fillna(0)

    logging.info('One hot encode categorical features')
    df_train_data['test_train'] = 0
    df_test_data['test_train'] = 1
    df_test_train = df_train_data.append(df_test_data)
    df_test_train = pd.get_dummies(df_test_train)

    logging.info('Check for correlated features and drop if needed')
    obj_drop_cols = obj_preds.check_corr_features(
            df_test_train.drop(columns={'test_train'}), args.cut_val)
    if len(obj_drop_cols) > 0:
        logging.info(f'dropping correlated cols: {obj_drop_cols}')
        df_test_train = df_test_train.drop(columns=obj_drop_cols)

    logging.info('Split data back into test and train sets')
    df_train_data = df_test_train[df_test_train['test_train'] == 0]
    df_train_data = df_train_data.drop(columns={'test_train'})
    df_test_data = df_test_train[df_test_train['test_train'] == 1]
    df_test_data = df_test_data.drop(columns={'test_train'})

    logging.info('''Run test train split on train data and
                 find accuracy of selected model''')
    X_train, X_test, y_train, y_test = train_test_split(df_train_data,
                                                        y_train,
                                                        test_size=0.30,
                                                        stratify=y_train)
    obj_preds.set_x_train(X_train)
    obj_preds.set_y_train(y_train)
    obj_preds.set_x_test(X_test)
    switcher = {'naive_bayes': obj_preds.run_naive_bayes,
                'random_forest': obj_preds.run_random_forest,
                'logit': obj_preds.run_logistic_regression}
    if args.str_ml_model is not None:
        switcher.get(args.str_ml_model)(bln_fit=True, bln_predict=True)
        acc_score = metrics.accuracy_score(y_test, obj_preds.get_preds())
        logging.info(f'accuracy score: {acc_score}')
    else:
        logging.info('Running all 3 models')
        acc_score = 0
        for key in switcher:
            switcher.get(key)(bln_fit=True, bln_predict=True)
            acc_score_tmp = metrics.accuracy_score(y_test,
                                                   obj_preds.get_preds())
            if acc_score_tmp > acc_score:
                acc_score = acc_score_tmp
                args.str_ml_model = key
        logging.info(f'''Using final model: {args.str_ml_model}
        with accuracy: {acc_score}''')

    logging.info('Running Final Predictions')
    obj_preds.set_x_test(df_test_data)
    switcher.get(args.str_ml_model)(bln_fit=False, bln_predict=True)
    obj_final_predictions = obj_preds.get_preds()
    df_preds_out = df_test_data[['passenger_id']]
    df_preds_out = df_preds_out.rename(columns={'passenger_id': 'PassengerId'})
    df_preds_out['Survived'] = obj_final_predictions
    df_preds_out.to_csv(args.str_out_file, index=False)


def main(passed_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--str_training_data', type=str,
                        default='./training.csv',
                        help='path to training data set')
    parser.add_argument('--str_test_data', type=str, default='./test.csv',
                        help='path to test data set')
    parser.add_argument('--str_logging', type=str, default='titanic.log',
                        help='file to output logging to')
    parser.add_argument('--str_ml_model', type=str, default=None,
                        help='model to run')
    parser.add_argument('--cut_val', type=float, default=0.9,
                        help='''remove columns with correlation greater than
                        or equal to this value''')
    parser.add_argument('--bln_run_grid_search', default=False,
                        action='store_true',
                        help='''If true, will run grid search
                        for the random forest model''')
    parser.add_argument('--param_grid', type=json.loads, default=None,
                        help='''If true, will run grid search
                        for the random forest model''')
    parser.add_argument('--str_out_file', type=str,
                        default='./final_titanic_survival_predictions.csv',
                        help='file path and name of final output file')
    args = parser.parse_args(passed_args)
    log_format = "%(asctime)s::%(levelname)s::"\
                 "%(filename)s::%(message)s"
    logging.basicConfig(filename=args.str_logging,
                        filemode='w', level='INFO',
                        format=log_format)

    run_titanic_preds(args)


if __name__ == '__main__':
    main()
