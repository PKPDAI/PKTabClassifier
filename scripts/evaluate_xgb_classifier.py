import json
import pickle
from pathlib import Path

import pandas as pd
import torch
import typer
from sentence_transformers import SentenceTransformer, models

from pk_tableclass.preprocessing import serialize_features, combine_features, prepare_table_features
from pk_tableclass.utils import print_model_scores


def main(
        path_to_config: Path =typer.Option("configs/config.json",
                                    help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        path_to_trained_model: Path = typer.Option(default="../trained_models/best_classifier.pkl", help="Path to trained model, expects .pkl file."),
        test_data_path: Path = typer.Option("data/test.pkl", help="Path to test data file, expects .pkl file.")
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    class_labels = args["class_labels"]
    label2id = {label: idx for idx, label in enumerate(class_labels)}
    id2label = {idx: label for idx, label in enumerate(class_labels)}
    feature_names = args["feature"]

    # ============= Read in and prepare data ================= #
    raw_test_data = pd.read_pickle(test_data_path)
    test_df = prepare_table_features(raw_test_data)

    # ============= Set device ================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("===========================\n")

    # ============= Load model ================= #
    classifier = pickle.load(open(path_to_trained_model, 'rb'))

    # ============= Prep Labels ================= #
    y_test = [label2id[label] for label in test_df["label"].to_list()]

    # ============= Prep Features ================= #
    if args["representation_method"] == "bow":
        X_test = serialize_features(feature_names, test_df)

    else:
        test_features = combine_features(feature_names, test_df, args["representation_method"], chunking=True)
        word_embedding_model = models.Transformer(args["representation_method"])
        pooling_model = models.Pooling(
            word_embedding_model.get_word_embedding_dimension(),
            pooling_mode=args["pooling_mode"]
        )
        model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

        X_test = model.encode(test_features, batch_size=32, device=device,
                                      show_progress_bar=True, output_value='sentence_embedding')

    # ============= Get Predictions and Scores ================= #
    y_test_pred = classifier.predict(X_test)
    print_model_scores(y_labels=y_test, y_preds=y_test_pred, id2label=id2label, condition_name="Best Classifier")

if __name__ == "__main__":
    typer.run(main)

