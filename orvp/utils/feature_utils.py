"""
Author: Mrutyunjay Biswal
Project: Optiver Realized Volatility Prediction Challenge, Kaggle 2021

This file consists of utilities functions for feature engineering on Order book data. The data comes from
book_train/test.parquet, and is linked with train.csv by stock id. Attributes of the data:

1. time id
2. seconds_in_bucket
3. bid_price1
4. ask_price1
5. bid_price2
6. ask_price2
7. bid_size1
8. ask_size1
9. bid_size2
10. ask_size2

For more details on the dataset, refer to the Readme and the competition page.
"""

import numpy as np
import pandas as pd


def calculate_wap(df, rank="1"):
    """
    Weighted Average Pricing for a stock at a given time ID is given by:
    (bid_price1 * ask_size1 + bid_size1 * ask_price1)/(bid_size1 + ask_size1)

    It can further be extended to:

        sum(bid_price_i * ask_size_i + bid_size_i * ask_price_i)/sum(bid_size_i + ask_size_i)

    :param rank: which wap to calculate
    :param df: parquet table containing order book
    :return:
    """
    return (df[f"bid_price{rank}"] * df[f"ask_size{rank}"] + df[f"bid_size{rank}"] * df[f"ask_price{rank}"]) / (df[f"bid_size{rank}"] + df[f"ask_size{rank}"])


def calculate_inter_wap(df, rank="1"):
    return (df[f"bid_price{rank}"] * df[f"bid_size{rank}"] + df[f"ask_size{rank}"] * df[f"ask_price{rank}"]) / (
                df[f"bid_size{rank}"] + df[f"ask_size{rank}"])
    pass


def calculate_log_return(series):
    return np.log(series).diff()


def calculate_rv(series):
    return np.sqrt(np.sum(np.square(series)))


def count_unique(series):
    return len(np.unique(series))


def get_stats_window(df, seconds_in_bucket, features_dict, add_suffix=False):
    df_feature = df[df["seconds_in_bucket"] >= seconds_in_bucket].groupby(["time_id"]).agg(features_dict).reset_index()
    df_feature.columns = ["_".join(col) for col in df_feature.columns]

    if add_suffix:
        df_feature = df_feature.add_suffix("_" + str(seconds_in_bucket))

    return df_feature
    pass


def window_stats(df, feature_dict, second_windows):
    df_merged = get_stats_window(df, seconds_in_bucket=0, features_dict=feature_dict)

    temp_dfs = []
    for window in second_windows:
        temp_dfs.append((window, get_stats_window(df, seconds_in_bucket=window, features_dict=feature_dict, add_suffix=True)))

    for window, temp_df in temp_dfs:
        df_merged = df_merged.merge(temp_df, how="left", left_on="time_id_", right_on=f"time_id__{window}")
        df_merged.drop(columns=[f"time_id__{window}"], inplace=True)

    return df_merged
    pass
