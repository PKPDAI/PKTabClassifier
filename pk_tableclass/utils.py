from pathlib import Path
from typing import Iterable, List, Dict

import numpy as np
import pandas as pd
import ujson
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, f1_score


def read_jsonl(file_path: Path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding='utf-8') as f:
        for line in f:
            try:
                yield ujson.loads(line.strip())
            except ValueError:
                continue

def write_jsonl(file_path: Path, lines: Iterable):
    # Taken from prodigy
    """Create a .jsonl file and dump contents.
    file_path (unicode / Path): The path to the output file.
    lines (list): The JSON-serializable contents of each line.
    """
    data = [ujson.dumps(line, escape_forward_slashes=False) for line in lines]
    Path(file_path).open('w', encoding='utf-8').write('\n'.join(data))


def convert_label2id_to_id2label(label2id):
    """
    Convert a label2id dictionary to an id2label dictionary.

    Parameters:
    label2id (dict): A dictionary mapping labels to IDs.

    Returns:
    dict: A dictionary mapping IDs to labels.
    """
    id2label = {idx: label for label, idx in label2id.items()}
    return id2label


def print_model_scores(
        y_labels: List[int],
        y_preds: List[int],
        id2label: Dict[int, str],
        condition_name: str
) -> None:
    """
    Prints a classification report for the given test and predicted labels, formatted with percentages
    rounded to two decimal places.

    Parameters:
    - y_test (List[int]): True labels of the test set.
    - y_test_pred (List[int]): Predicted labels from the model.
    - id2label (Dict[int, str]): A dictionary mapping label indices to their string names.
    - condition_name (str): The name of the condition being evaluated.

    Returns:
    - None: Outputs the classification report directly to the console.
    """
    print(f"Classification report (%) to 2 d.p for {condition_name}")
    cr = classification_report(
        y_labels,
        y_preds,
        target_names=[id2label[i] for i in range(len(id2label))],
        output_dict=True
    )
    df = pd.DataFrame(cr).transpose()
    metrics_columns = ['precision', 'recall', 'f1-score']
    df[metrics_columns] = df[metrics_columns] * 100
    df_rounded = df.round(2)
    print(df_rounded)

def make_confidence_histogram(
    y_confidences: List[float],
    condition_name: str
) -> None:
    """
    Creates and saves a histogram of confidence scores for the given condition.

    Parameters:
    - y_test_confidence (List[float]): A list of confidence scores for the test predictions.
    - condition_name (str): The name of the condition being evaluated. Used in the title and filename.

    Returns:
    - None: Displays the histogram plot and saves it as a PNG file.
    """
    plt.hist(y_confidences, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.savefig(f"../figures/hist_conf_scores_{condition_name}.png")
    plt.title(f"Distribution of Confidence Scores for {condition_name}")
    plt.show()


def make_confidence_plots(
    y_confidences: List[float],
    y_preds: List[int],
    y_labels: List[int],
    condition_name: str
) -> None:
    """
    Creates and saves plots for analyzing the relationship between confidence scores and model performance.

    The function generates:
    1. A histogram showing the distribution of confidence scores.
    2. A plot showing the F1 score and proportion of the dataset below varying confidence thresholds.

    Parameters:
    - y_test_confidence (List[float]): A list of confidence scores for the test predictions.
    - y_test_pred (List[int]): A list of predicted labels from the model.
    - y_test (List[int]): A list of true labels of the test set.
    - condition_name (str): The name of the condition being evaluated, used in titles and filenames.

    Returns:
    - None: Outputs the plots directly and saves them as PNG files.
    """

    # 1. Plot histogram of confidence distributions in dataset
    plt.hist(y_confidences, bins=20, edgecolor='k', alpha=0.7)
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.savefig(f"../figures/hist_conf_scores_{condition_name}.png")
    plt.title(f"Distribution of Confidence Scores for {condition_name}")
    plt.show()


    # 2. Plot performance across a range of confidence thresholds
    thresholds = np.arange(0.5, 1.05, 0.05)
    f1_scores_below = []
    proportions_below_threshold = []

    for threshold in thresholds:
        below_threshold_mask = y_confidences < threshold

        if np.sum(below_threshold_mask) > 0:
            f1_below = f1_score(np.array(y_labels)[below_threshold_mask], y_preds[below_threshold_mask],
                                average='weighted')
        else:
            f1_below = 0

        proportion_below = np.mean(below_threshold_mask)
        f1_scores_below.append(f1_below)
        proportions_below_threshold.append(proportion_below)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1_scores_below, label="F1 Score (below threshold)")
    plt.plot(thresholds, proportions_below_threshold, label="Proportion of Dataset Below Threshold")
    plt.xlabel("Confidence Threshold")
    plt.legend()
    plt.savefig(f"../figures/conf_scores_f1_vs_data_{condition_name}.png")
    plt.title(f"F1 Score and Proportion Below Confidence Threshold for {condition_name}")
    plt.show()
