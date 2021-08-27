import gc
import sys
from orvp.dataset.create_data import *
from orvp.modeling.lgb_models import *
from orvp.config import ConFig as cfg

if __name__ == "__main__":
    _ = gc.collect()
    is_train = False

    model = TrainFer(cfg.model_params["lgb_bl"], n_splits=5, model_path=cfg.paths["lgb_baseline"])

    if is_train:
        train = pd.read_csv(cfg.paths["train_csv"])
        train_data = GetData(train, cfg.paths["train_book"], cfg.paths["train_trade"])
        train_df = train_data.get_features()

        model.train(train_df.drop(columns=["row_id", "target", "time_id"]), train_df["target"])
    else:
        test = pd.read_csv(cfg.paths["test_csv"])
        test_data = GetData(test, cfg.paths["test_book"], cfg.paths["test_trade"])
        test_df = test_data.get_features()

        preds = model.infer(test_df.drop(columns=["row_id", "time_id"]))
        test["target"] = preds
        test[["row_id", "target"]].to_csv("./submission.csv", index=False)
        print(test.head())
