import os
from tqdm import tqdm
from joblib import Parallel, delayed

from ..utils.order_book_features import *
from ..utils.trade_features import *

tqdm.pandas()


class GetData:
    def __init__(self, df, book_path, trade_path):
        self.df = df.copy(deep=True)
        self.order_book_path = book_path
        self.trade_path = trade_path

        self._get_rowid()

    def _get_rowid(self):
        self.df["row_id"] = self.df["stock_id"].astype(str) + "-" + self.df["time_id"].astype(str)

    def get_time_stock(self, buck_windows=cfg.bucket_windows):
        vol_cols = []
        feat_set = ['log_return1_calculate_rv', 'log_return2_calculate_rv', 'trade_log_return_calculate_rv']
        for feat in feat_set:
            for sec in buck_windows:
                vol_cols.append(feat + f'_{sec}')
        vol_cols += feat_set

        df_stock_id = self.df.groupby(['stock_id'])[vol_cols].agg(['mean', 'std', 'max', 'min']).reset_index()
        df_stock_id.columns = ['_'.join(col) for col in df_stock_id.columns]
        df_stock_id = df_stock_id.add_suffix('_' + 'stock')

        df_time_id = self.df.groupby(['time_id'])[vol_cols].agg(['mean', 'std', 'max', 'min']).reset_index()
        df_time_id.columns = ['_'.join(col) for col in df_time_id.columns]
        df_time_id = df_time_id.add_suffix('_' + 'time')

        # Merge with original dataframe
        self.df = self.df.merge(df_stock_id, how='left', left_on=['stock_id'], right_on=['stock_id__stock'])
        self.df = self.df.merge(df_time_id, how='left', left_on=['time_id'], right_on=['time_id__time'])
        self.df.drop(['stock_id__stock', 'time_id__time'], axis=1, inplace=True)
        return self.df

    def process_features(self, list_stock_ids):
        def parallel_helper(stock_id):
            book_sample_path = os.path.join(self.order_book_path, f"stock_id={stock_id}")
            trade_sample_path = os.path.join(self.trade_path, f"stock_id={stock_id}")

            return pd.merge(get_book_features(book_sample_path), get_trade_features(trade_sample_path),
                            on="row_id",
                            how="left")

        df = Parallel(n_jobs=4, verbose=1)(delayed(parallel_helper)(stock_id) for stock_id in list_stock_ids)
        df = pd.concat(df, ignore_index=True)

        return df

    def _get_features(self):
        features_df = self.process_features(self.df["stock_id"].unique())
        self.df = self.df.merge(features_df, on=["row_id"], how="left")

        return self.get_time_stock()
        pass

    def get_all_features(self, stock_groups):
        return create_cluster_aggregations(self._get_features(), stock_groups)
        pass
