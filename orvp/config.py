import numpy as np

from .utils.feature_utils import calculate_rv, count_unique


class ConFig:
    
    paths = {
        # train path
        "train_csv": "./inp/Raw Data/train.csv",
        "train_book": "./inp/Raw Data/book_train.parquet",
        "train_trade": "./inp/Raw Data/trade_train.parquet",

        # test path
        "test_csv": "./inp/Raw Data/test.csv",
        "test_book": "./inp/Raw Data/book_test.parquet",
        "test_trade": "./inp/Raw Data/trade_test.parquet",

        # model paths
        "xgb_baseline": "./models/xgbBaseline/"
    }

    feature_dict_book = {
        "wap1": [np.sum, np.mean, np.std],
        "wap2": [np.sum, np.mean, np.std],
        "iwap1": [np.sum, np.mean, np.std],
        "iwap2": [np.sum, np.mean, np.std],
        "log_return1": [np.sum, calculate_rv, np.mean, np.std],
        "log_return2": [np.sum, calculate_rv, np.mean, np.std],
        "inter_log_return1": [np.sum, calculate_rv, np.mean, np.std],
        "inter_log_return2": [np.sum, calculate_rv, np.mean, np.std],
        "wap_balance": [np.sum, np.mean, np.std],
        "volume_imbalance": [np.sum, np.mean, np.std],
        "total_volume": [np.sum, np.mean, np.std],
        "price_spread1": [np.sum, np.mean, np.std],
        "price_spread2": [np.sum, np.mean, np.std],
        "bid_spread": [np.sum, np.mean, np.std],
        "ask_spread": [np.sum, np.mean, np.std],
    }

    feature_dict_trade = {
        "log_return": [calculate_rv],
        "seconds_in_bucket": [count_unique],
        "size": [np.sum],
        "order_count": [np.mean, np.sum],
        "amount": [np.mean, np.sum, np.std]
    }

    model_params = {
        "xgb_bl": {
            "objective": "reg:squarederror",
            "booster": "gbtree",
            "nthread": -1,
            "eta": 0.3,
            "max_depth": 8,
            "min_child_weight": 1,
            "sampling_method": "uniform",
            # "tree_method": "gpu_hist"
        },
        "lgb_bl": {
            "objective": "rmse",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
        }
    }

    bucket_windows = [100, 200, 300, 400, 500]
    random_state = 2021
    pass
