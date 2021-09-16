from .feature_utils import *
from ..config import ConFig as cfg


def get_trade_price_features(df):
    res = []
    for n_time_id in df['time_id'].unique():
        df_id = df[df['time_id'] == n_time_id]
        vol_tendency = tendency(df_id['price'].values, df_id['size'].values)
        f_max = np.sum(df_id['price'].values > np.mean(df_id['price'].values))
        f_min = np.sum(df_id['price'].values < np.mean(df_id['price'].values))
        df_max = np.sum(np.diff(df_id['price'].values) > 0)
        df_min = np.sum(np.diff(df_id['price'].values) < 0)
        abs_diff = np.median(np.abs(df_id['price'].values - np.mean(df_id['price'].values)))
        energy = np.mean(df_id['price'].values ** 2)
        iqr_p = np.percentile(df_id['price'].values, 75) - np.percentile(df_id['price'].values, 25)
        abs_diff_v = np.median(np.abs(df_id['size'].values - np.mean(df_id['size'].values)))
        energy_v = np.sum(df_id['size'].values ** 2)
        iqr_p_v = np.percentile(df_id['size'].values, 75) - np.percentile(df_id['size'].values, 25)

        res.append({'time_id': n_time_id,
                    'tendency': vol_tendency,
                    'f_max': f_max,
                    'f_min': f_min,
                    'df_max': df_max,
                    'df_min': df_min,
                    'abs_diff': abs_diff,
                    'energy': energy,
                    'iqr_p': iqr_p,
                    'abs_diff_v': abs_diff_v,
                    'energy_v': energy_v,
                    'iqr_p_v': iqr_p_v})

    return pd.DataFrame(res)
    pass


def tau_features(df, sec, weight):
    tau_feat = 'tau_' + str(sec)
    bucket_col = 'trade_seconds_in_bucket_count_unique_' + str(sec)
    df[tau_feat] = np.sqrt(weight/df[bucket_col])

    size_feat = 'size_' + str(sec)
    order_col = 'trade_order_count_sum_' + str(sec)
    df[size_feat] = np.sqrt(weight/df[order_col])

    return df
    pass


def get_trade_features(file_path, buck_windows=cfg.bucket_windows):
    trade_df = pd.read_parquet(file_path)

    trade_df["log_return"] = trade_df.groupby(["time_id"])["price"].apply(calculate_log_return)
    trade_df["amount"] = trade_df["size"] * trade_df["price"]

    price_features = get_trade_price_features(trade_df)
    trade_df_merged = window_stats(trade_df, cfg.feature_dict_trade, buck_windows, price_features)

    trade_df_merged = trade_df_merged.add_prefix("trade_")

    trade_df_merged["row_id"] = trade_df_merged["trade_time_id_"].apply(lambda x: f"{file_path.split('=')[1]}-{x}")
    trade_df_merged.drop(["trade_time_id_"], axis=1, inplace=True)

    for sec in buck_windows:
        trade_df_merged = tau_features(trade_df_merged, sec, weight=sec/600)
    return trade_df_merged.bfill().ffill()
