import json
from pathlib import Path

import pandas as pd
import torch
import typer
import xgboost as xgb
from sentence_transformers import SentenceTransformer, models
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.utils import compute_sample_weight

from pk_tableclass.preprocessing import serialize_features, combine_features, prepare_table_features


def main(
        path_to_config: Path = typer.Option("configs/config.json",
                                            help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        train_data_path: Path = typer.Option("data/train.pkl",
                                             help="Path to training data file, expects .pkl file."),
        val_data_path: Path = typer.Option("data/validation.pkl",
                                           help="Path to validation data file, expects .pkl file."),
        k: int = typer.Option(default=10, help="Number of folds for cross-validation."),
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

    # ============= Define the Classifier ================= #
    xgb_clf = xgb.XGBClassifier(learning_rate=0.1,
                                n_estimators=150,
                                num_class=3,
                                objective="multi:softmax",
                                nthread=4,
                                seed=args["seed"])

    # ============= Test Representation Methods ================= #
    representation_methods = [
        "bert-base-uncased",
        "sentence-transformers/all-mpnet-base-v2",
        'intfloat/e5-small-v2',
        "dmis-lab/biobert-v1.1",
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"
        # bow
    ]

    for representation_method in representation_methods:
        print("\n=============================\n")
        print(f"Training XGBoost with representations from {representation_method}")
        print(f"--------------------------\n")
        if representation_method == "bow":
            vectorizer = CountVectorizer()
            train_features = serialize_features(feature_names, train_df)
            val_features = serialize_features(feature_names, val_df)
            X_train = vectorizer.fit_transform(train_features)
            X_val = vectorizer.transform(val_features)

        else:
            train_features = combine_features(feature_names, train_df, representation_method, chunking=True)
            val_features = combine_features(feature_names, val_df, representation_method, chunking=True)
            word_embedding_model = models.Transformer(representation_method)
            pooling_model = models.Pooling(
                word_embedding_model.get_word_embedding_dimension(),
                pooling_mode=args["pooling_mode"]
            )
            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

            X_train = model.encode(train_features, batch_size=32, device=device,
                                          show_progress_bar=True, output_value='sentence_embedding')
            X_val = model.encode(val_features, batch_size=32, device=device, show_progress_bar=True,
                                        output_value='sentence_embedding')



        # ============= Fit & Save the Classifier ================= #
        xgb_clf.fit(X_train, y_train, sample_weight=class_weights)

        # ============= Validation metrics ================= #
        print("\n=============Validation =============.\n")
        y_val_pred = xgb_clf.predict(X_val)
        class_report = classification_report(y_val, y_val_pred, target_names=[id2label[i] for i in range(len(id2label))])
        print('\nClassification Report:')
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

        cv_scores = cross_validate(xgb_clf, X_train, y_train, cv=cv, scoring=cv_scoring,
                                   params={"sample_weight": class_weights})
        formatted_cv_scores = {k: (round(v.mean() * 100, 2), round(v.std() * 100, 2)) for k, v in cv_scores.items()}
        print(formatted_cv_scores)

        print("===========================\n")


if __name__ == "__main__":
    typer.run(main)