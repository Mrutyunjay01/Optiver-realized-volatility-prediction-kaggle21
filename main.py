import sys
from src.dataset.create_data import *
from src.modeling.xgb_models import *
from src.config import ConFig as cfg


if __name__ == '__main__':
    is_train = sys.argv[1]

    if is_train:
        train = pd.read_csv(cfg.paths["train_csv"])
        train_data = GetData(train, cfg.paths["train_book"], cfg.paths["train_trade"])
        train_df = train_data.get_features()
        print(train_df.head())

        trainer = TrainFer(cfg.model_params["xgb_bl"], n_splits=5, model_path=cfg.paths["xgb_baseline"])
        trainer.train(train_df.drop(columns=["row_id", "target", "time_id"]), train_df["target"])
    
    else:
        test = pd.read_csv(cfg.paths["test_csv"])
        test_data = GetData(test, cfg.paths["test_book"], cfg.paths["test_trade"])
        test_df = test_data.get_features()
        print(test_df.head())

        tester = TrainFer(cfg.model_params["xgb_bl"], n_splits=5, model_path=cfg.paths["xgb_baseline"])
        preds = tester.infer(test_df.drop(columns=["row_id", "time_id"])) 

        test["target"] = preds
        print(test.head())
