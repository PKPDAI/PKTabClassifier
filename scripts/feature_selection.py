import json
from pathlib import Path

import pandas as pd
import torch
import typer
import xgboost as xgb
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_sample_weight
from tqdm import tqdm

from pk_tableclass.preprocessing import prepare_table_features, serialize_features


def main(
        path_to_config: Path = typer.Option("configs/config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        train_data_path: Path = typer.Option("data/train.pkl",
                                             help="Path to training data file, expects .pkl file."),
        val_data_path: Path = typer.Option("data/validation.pkl",
                                           help="Path to validation data file, expects .pkl file."),
        save_results: bool = typer.Option(default=False, help="Whether to save results to .json file"),
        results_save_path: Path = typer.Option(default="data/PK_tableclass_results_features.jsonl"),
        k: int = typer.Option(default=10, help="Number of folds for cross-validation."),
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    class_labels = args["class_labels"]
    label2id = {label: idx for idx, label in enumerate(class_labels)}
    features = [["caption"], ["first_column"], ["header_row"], ["first_few_rows"], ["markdown"], ["footer"],
                ["caption", "markdown"]]

    # ============= Read in and prepare data ================= #
    raw_train_data = pd.read_pickle(train_data_path)
    raw_val_data = pd.read_pickle(val_data_path)
    train_df = prepare_table_features(raw_train_data)
    val_df = prepare_table_features(raw_val_data)

    print("=============Dataset Statistics==============\n")
    print(train_df["label"].value_counts())
    print("\nValidation Data:")
    print(val_df["label"].value_counts())
    print("===========================\n")

    # ============= Set device ================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("===========================\n")

    # ============= Prep Labels ================= #
    y_train = [label2id[label] for label in train_df["label"].to_list()]
    class_weights = compute_sample_weight('balanced', y_train)
    y_val = [label2id[label] for label in val_df["label"].to_list()]

    # ============= Define the Classifier ================= #
    vectorizer_count = CountVectorizer()
    xgb_clf = xgb.XGBClassifier(learning_rate=0.1,
                                n_estimators=150,
                                num_class=3,
                                objective="multi:softmax",
                                nthread=4,
                                seed=args["seed"])

    xgb_pipe = Pipeline([
        ('vectorizer', vectorizer_count),
        ('xgbclassifier', xgb_clf)
    ])

    # ============= Prep Features ================= #
    for f in tqdm(features):
        print("\n=============================\n")
        print(f"Training XGBoost on model Embeddings of {f}")
        print(f"--------------------------\n")
        X_train = serialize_features(f, train_df)
        X_val = serialize_features(f, val_df)

        #Evaluate the model on validation set
        xgb_pipe.fit(X_train, y_train, xgbclassifier__sample_weight=class_weights)
        y_val_pred = xgb_pipe.predict(X_val)
        class_report = classification_report(y_val, y_val_pred)

        # Print evaluation results
        print('Classification Report:')
        print(class_report)

        # cross validations
        cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=args["seed"])
        cv_scoring = {'prec_macro': 'precision_macro',
                      'rec_macro': "recall_macro",
                      "f1_macro": "f1_macro",
                      "f1_micro": "f1_micro",
                      "prec_micro": "precision_micro",
                      "rec_micro": "recall_micro",
                      "f1_weighted": "f1_weighted",
                      "prec_weighted": "precision_weighted",
                      "rec_weighted": "recall_weighted"
                      }
        cv_scores = cross_validate(xgb_pipe, X_train, y_train, cv=cv, scoring=cv_scoring,
                                   fit_params={"xgbclassifier__sample_weight": class_weights})
        formatted_cv_scores = {k: (round(v.mean() * 100, 2), round(v.std() * 100, 2)) for k, v in cv_scores.items()}
        print(formatted_cv_scores)

        if save_results:
            results_dict = {
                "feature": str(f),
                "class_weighting": "yes",
                "encoder_model": "bow",
                "class_model": "xgb",
                "CV_scores": formatted_cv_scores,
                "cv_k": k,
            }

            with open(results_save_path, 'a') as file:
                file.write(json.dumps(results_dict) + '\n')

    print("==========Finished===========")

if __name__ == "__main__":
    typer.run(main)