import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

import xgboost as xgb

from ..utils.metric_utils import *


class TrainFer:
    def __init__(self, params_dict, n_splits, model_path, random_state=2021):
        self.params = params_dict
        self.n_splits = n_splits
        self.random_state = random_state
        self.model_path = model_path

        if not os.path.isdir(model_path):
            os.makedirs(model_path)

    def train(self, X, y):
        oof_predictions = np.zeros(X.shape[0])
        kfold = KFold(n_splits=self.n_splits, random_state=self.random_state, shuffle=True)

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold - {fold}\n")

            x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            dtrain = xgb.DMatrix(x_train, label=y_train, weight=1 / np.square(y_train), enable_categorical=True)
            dval = xgb.DMatrix(x_val, label=y_val, weight=1 / np.square(y_val), enable_categorical=True)

            model = xgb.train(self.params,
                              dtrain=dtrain,
                              num_boost_round=1000,
                              evals=[(dtrain, "dtrain"), (dval, "dval")],
                              verbose_eval=10,
                              feval=feval_rmspe,
                              early_stopping_rounds=200)

            pickle.dump(model, open(os.path.join(self.model_path, f"xgb_bl_{fold}.pkl"), "wb"))
            oof_predictions[val_idx] = model.predict(dval)

        rmspe_score = rmspe(y, oof_predictions)
        print(f"OOF RMSPE: {rmspe_score}")

    def infer(self, x_test):

        test_predictions = np.zeros(x_test.shape[0])
        dtest = xgb.DMatrix(x_test)

        for mpth in os.listdir(self.model_path):
            model = pickle.load(open(os.path.join(self.model_path, mpth), "rb"))
            test_predictions += model.predict(dtest) / 5

        return test_predictions
        pass
