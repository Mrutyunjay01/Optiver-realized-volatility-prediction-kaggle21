from .feature_utils import *
from ..config import ConFig as cfg


def get_trade_features(file_path):
    trade_df = pd.read_parquet(file_path)

    trade_df["log_return"] = trade_df.groupby(["time_id"])["price"].apply(calculate_logreturn)

    trade_df_merged = get_stats_window(trade_df, seconds_in_bucket=0, features_dict=cfg.feature_dict_trade)

    trade_df_450 = get_stats_window(trade_df, seconds_in_bucket=450, features_dict=cfg.feature_dict_trade,
                                    add_suffix=True)
    trade_df_300 = get_stats_window(trade_df, seconds_in_bucket=300, features_dict=cfg.feature_dict_trade,
                                    add_suffix=True)
    trade_df_150 = get_stats_window(trade_df, seconds_in_bucket=150, features_dict=cfg.feature_dict_trade,
                                    add_suffix=True)

    # merge stats
    trade_df_merged = trade_df_merged.merge(trade_df_450, how="left", left_on="time_id_", right_on="time_id__450")
    trade_df_merged = trade_df_merged.merge(trade_df_300, how="left", left_on="time_id_", right_on="time_id__300")
    trade_df_merged = trade_df_merged.merge(trade_df_150, how="left", left_on="time_id_", right_on="time_id__150")

    trade_df_merged.drop(columns=["time_id__450", "time_id__300", "time_id__150"], inplace=True)

    trade_df_merged = trade_df_merged.add_prefix("trade_")

    trade_df_merged["row_id"] = trade_df_merged["trade_time_id_"].apply(lambda x: f"{file_path.split('=')[1]}-{x}")
    trade_df_merged.drop(["trade_time_id_"], axis=1, inplace=True)

    return trade_df_merged
