import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer

from pk_tableclass.preprocessing import serialize_features, prepare_table_features
from pk_tableclass.utils import print_model_scores, make_confidence_plots


def main(
        path_to_config: Path =typer.Option("configs/config.json",
                                    help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        val_data_path: Path =typer.Option("data/validation.pkl", help="Path to val data file, expects .pkl file."),
        path_to_trained_model: Path =typer.Option(default="trained_models/best_classifier.pkl",
                                           help="Path to trained model, expects .pkl file."),
):
    # ============= Read in & prep data ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    class_labels = args["class_labels"]
    label2id = {label: idx for idx, label in enumerate(class_labels)}
    id2label = {idx: label for idx, label in enumerate(class_labels)}
    feature_names = args["feature"]

    raw_val_data = pd.read_pickle(val_data_path)
    val_df = prepare_table_features(raw_val_data)
    X_val = serialize_features(feature_names, val_df)
    y_val = val_df["label"].to_list()
    y_val = [label2id[x] for x in y_val]

    # ============= Plot Confidence for Best Classifier ================= #
    best_classifier = pickle.load(open(path_to_trained_model, 'rb'))
    y_val_probs = best_classifier.predict_proba(X_val)
    y_val_confidence = np.max(y_val_probs, axis=1)
    y_val_pred = np.argmax(y_val_probs, axis=1)
    print_model_scores(y_val, y_val_pred, id2label, "Uncalibrated Classifier")
    make_confidence_plots(y_confidences=y_val_confidence, y_preds=y_val_pred, y_labels=y_val, condition_name="Uncalibrated Classifier")

if __name__ == '__main__':
    typer.run(main)