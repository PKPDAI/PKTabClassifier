import json
from pathlib import Path

import pandas as pd
import torch
import typer
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_sample_weight

from pk_tableclass.preprocessing import serialize_features, prepare_table_features


def main(
        path_to_config: Path = typer.Option("configs/config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        train_data_path: Path = typer.Option("data/train.pkl",
                                             help="Path to training data file, expects .pkl file."),
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    class_labels = args["class_labels"]
    label2id = {label: idx for idx, label in enumerate(class_labels)}
    feature_names = args["feature"]

    # ============= Read in and prepare data ================= #
    raw_train_data = pd.read_pickle(train_data_path)
    train_df = prepare_table_features(raw_train_data)

    # ============= Set device ================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("===========================\n")

    # ============= Prep Labels ================= #
    y_train = [label2id[label] for label in train_df["label"].to_list()]
    class_weights = compute_sample_weight('balanced', y_train)

    # ============= Prep Features ================= #
    vectorizer = CountVectorizer()
    train_features = serialize_features(feature_names, train_df)


    # ============= Define the Model ================= #
    clf = xgb.XGBClassifier(learning_rate=0.1,
                            n_estimators=150,
                            objective="multi:softmax",
                            num_class=3,
                            nthread=4,
                            seed=1)

    xgb_pipe = Pipeline([
        ('vectorizer', vectorizer),
        ('xgbclassifier', clf)
    ])
    # ============= Define the Hyperparameters ================= #
    params = {
        'vectorizer__max_features': [1000, 2000, 3000, 4000],
        "xgbclassifier__max_depth": range(2, 10, 2),
        "xgbclassifier__min_child_weight": range(1, 6, 2),
        "xgbclassifier__gamma": [i/10.0 for i in range(0,5)],
        "xgbclassifier__subsample": [i/10.0 for i in range(5,10)],
        "xgbclassifier__colsample_bytree":[i/10.0 for i in range(3,10)],
        'xgbclassifier__reg_alpha':[0, 1e-5, 1e-2, 0.1, 1],
        'xgbclassifier__reg_lambda':[0, 1e-5, 1e-2, 0.1, 1],
    }

    # ============= Perform Grid Search ================= #
    gsearch = GridSearchCV(estimator=xgb_pipe, param_grid=params, scoring='f1_macro', cv=5, verbose=5)
    gsearch.fit(train_features, y_train, xgbclassifier__sample_weight=class_weights)

    print('\n Best estimator:')
    print(gsearch.best_estimator_)
    print('\n Best hyperparameters:')
    print(gsearch.best_params_)

if __name__ == "__main__":
    typer.run(main)