import json
import pickle
from pathlib import Path

import pandas as pd
import torch
import typer
import xgboost as xgb
from sentence_transformers import SentenceTransformer, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_sample_weight

from pk_tableclass.preprocessing import serialize_features, combine_features, prepare_table_features


def main(
        path_to_config: Path = typer.Option("configs/config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        train_data_path: Path = typer.Option("data/train.pkl",
                                             help="Path to training data file, expects .pkl file."),
        val_data_path: Path = typer.Option("data/validation.pkl",
                                           help="Path to validation data file, expects .pkl file."),
        model_save_dir: Path = typer.Option("trained_models/",
                                            help="Directory to save trained model to."),
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    class_labels = args["class_labels"]
    label2id = {label: idx for idx, label in enumerate(class_labels)}
    id2label = {idx: label for idx, label in enumerate(class_labels)}
    feature_names = args["feature"]

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


    # ============= Prep Features ================= #
    if args["representation_method"] == "bow":
        vectorizer = CountVectorizer(max_features=args["bow_max_features"])
        train_features = serialize_features(feature_names, train_df)
        val_features = serialize_features(feature_names, val_df)
        X_train = vectorizer.fit_transform(train_features)
        X_val = vectorizer.transform(val_features)

    else:
        train_features = combine_features(feature_names, train_df, args["representation_method"], chunking=True)
        val_features = combine_features(feature_names, val_df, args["representation_method"], chunking=True)
        word_embedding_model = models.Transformer(args["representation_method"])
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=args["pooling_mode"]
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        X_train = model.encode(train_features, batch_size=32, device=device,
                                      show_progress_bar=True, output_value='sentence_embedding')
        X_val = model.encode(val_features, batch_size=32, device=device, show_progress_bar=True,
                                    output_value='sentence_embedding')

    # ============= Define the Classifier ================= #
    xgb_early_stopping = xgb.XGBClassifier(
        learning_rate=args["classifier_lr"],
        max_depth=args["classifier_max_depth"],
        min_child_weight=args["classifier_min_child_weight"],
        gamma=args["classifier_gamma"],
        subsample=args["classifier_subsample"],
        colsample_bytree=args["classifier_colsample_bytree"],
        n_estimators=args["classifier_n_estimators"],
        num_class=args["classifier_num_classes"],
        objective="multi:softmax",
        nthread=4,
        seed=args["seed"],
        eval_metric=args["early_stopping_eval_metric"],
        early_stopping_rounds=args["early_stopping_rounds"],
    )

    # ============= Fit & Save the Classifier ================= #
    xgb_early_stopping.fit(X_train, y_train, sample_weight=class_weights,
                           eval_set=[(X_val, y_val)])

    if args["representation_method"] == "bow":
        best_pipe = Pipeline([
            ('vectorizer', vectorizer),
            ('xgbclassifier', xgb_early_stopping)
        ])
    else:
        best_pipe = xgb_early_stopping

    model_save_path = str(model_save_dir) + "/best_classifier.pkl"
    pickle.dump(best_pipe, open(model_save_path, 'wb'))

    # ============= Validation metrics ================= #
    y_val_pred = xgb_early_stopping.predict(X_val)
    print("\n=============Validation Scores=============.\n")
    conf_matrix = confusion_matrix(y_val, y_val_pred)
    print('Confusion Matrix:')
    print(conf_matrix)
    class_report = classification_report(y_val, y_val_pred, target_names=[id2label[i] for i in range(len(id2label))])
    print('\nClassification Report:')
    print(class_report)
    print("===========================\n")


if __name__ == "__main__":
    typer.run(main)