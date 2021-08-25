from .feature_utils import *
from ..config import ConFig as cfg


def get_trade_features(file_path):
    trade_df = pd.read_parquet(file_path)

    trade_df["log_return"] = trade_df.groupby(["time_id"])["price"].apply(calculate_logreturn)

    trade_df_merged = window_stats(trade_df, cfg.feature_dict_trade, [450, 300, 150])

    trade_df_merged = trade_df_merged.add_prefix("trade_")

    trade_df_merged["row_id"] = trade_df_merged["trade_time_id_"].apply(lambda x: f"{file_path.split('=')[1]}-{x}")
    trade_df_merged.drop(["trade_time_id_"], axis=1, inplace=True)

    return trade_df_merged
