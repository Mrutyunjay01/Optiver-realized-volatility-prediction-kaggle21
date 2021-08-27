import os
import pickle

import numpy as np
from sklearn.model_selection import KFold

import lightgbm as lgb

from ..utils.metric_utils import feval_rmspe, rmspe


def feval_wrapper(y_pred, model):
    return feval_rmspe(y_pred, model, is_xgb=False)


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
        oof_scores = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
            print(f"\nFold - {fold}\n")

            x_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            x_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            dtrain = lgb.Dataset(x_train, y_train, weight=1 / np.square(y_train), categorical_feature=["stock_id"])
            dval = lgb.Dataset(x_val, y_val, weight=1 / np.square(y_val), categorical_feature=["stock_id"])

            model = lgb.train(params=self.params,
                              num_boost_round=10000,
                              train_set=dtrain,
                              valid_sets=dval,
                              verbose_eval=250,
                              early_stopping_rounds=200,
                              feval=feval_wrapper)

            pickle.dump(model, open(os.path.join(self.model_path, f"lgb_bl_{fold}.pkl"), "wb"))
            fold_preds = model.predict(x_val)
            oof_score = rmspe(y_val, fold_preds)
            print(f"\nRMSPE of fold {fold}: {oof_score}")

            oof_scores.append(oof_score)
            oof_predictions[val_idx] = fold_preds

        print(f"\nOOF Scores: {oof_scores}\n")
        rmspe_score = rmspe(y, oof_predictions)
        print(f"OOF RMSPE: {rmspe_score}")

    def infer(self, x_test):
        test_predictions = np.zeros(x_test.shape[0])

        for mpth in os.listdir(self.model_path):
            model = pickle.load(open(os.path.join(self.model_path, mpth), "rb"))
            test_predictions += model.predict(x_test) / 5

        return test_predictions
