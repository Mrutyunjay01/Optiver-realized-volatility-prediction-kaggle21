# Optiver Realized Volatility Prediction

Organizer: [Optiver](https://www.optiver.com/)  
Platform: [Kaggle](https://www.kaggle.com/)  
Competition Page: https://www.kaggle.com/c/optiver-realized-volatility-prediction/

## docs

- Related docs such as references, articles, documentation, etc to be saved in this directory.

## inp

```
- inp
|_ Raw Data
  |__ book_test.parquet
  |__ book_train.parquet
  |__ trade_test.parquet
  |__ trade_train.parquet
  |__ sample_submission.csv
  |__ train.csv
  |__ test.csv
 ```

## models

- Trained weights/pretrained-weights of various models used/referred in the solution.

## notebooks

- EDA, modelling, pipeline notebooks to be added here.

## src

- Code package, consisting of modularised code for data preparation, cross validation, feature engineering, modelling, inference code snippets.

# Instructions
For default set, make sure the dataset is downloaded to `inp/Raw Data` folder, and cross check the `src/config.py` file for paths reference.  
To Train on the data with default KFold split:  
`python main.py True`

To test:  
`python main.py False`

```
More to be added soon. Stay tuned, and feel free to provide suggestions via PR.
```