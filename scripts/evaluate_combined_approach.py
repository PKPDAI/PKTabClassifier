import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from pk_tableclass.postprocessing import process_llm_output, replace_values_at_indices
from pk_tableclass.preprocessing import serialize_features, prepare_table_features
from pk_tableclass.prompt_templates import cot_template
from pk_tableclass.utils import print_model_scores


def main(
        path_to_config: Path =typer.Option("configs/config.json",
                                    help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        test_data_path: Path =typer.Option("data/test.pkl", help="Path to test data file, expects .pkl file."),
        path_to_trained_model: Path =typer.Option(default="trained_models/iso_calibrated_best_classifier.pkl",
                                           help="Path to trained model, expects .pkl file."),
        confidence_threshold: float = typer.Option(default=0.90, help="Confidence threshold."),
):
    # ============= Read in data ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    class_labels = args["class_labels"]
    label2id = {label: idx for idx, label in enumerate(class_labels)}
    id2label = {idx: label for idx, label in enumerate(class_labels)}
    feature_names = args["feature"]

    raw_test_data = pd.read_pickle(test_data_path)
    test_df = prepare_table_features(raw_test_data)
    X_test = serialize_features(feature_names, test_df)
    y_test = test_df["label"].to_list()
    y_test = [label2id[x] for x in y_test]

    # ============= Load best classifier ================= #
    classifier = pickle.load(open(path_to_trained_model, 'rb'))
    probs = classifier.predict_proba(X_test)
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)

    below_threshold_indices = np.where(confidences < confidence_threshold)[0]
    double_check_df = test_df.iloc[below_threshold_indices]
    percentage_check = len(double_check_df) / len(test_df)
    print(f"Threshold: {confidence_threshold}, Percentage check: {percentage_check}.\n")

    double_check_markdown = double_check_df["markdown"].to_list()
    double_check_captions = double_check_df["caption"].to_list()

    # ============= Define LLM, prompt & chain ================= #
    os.environ['OPENAI_API_KEY'] = args["open_ai_api_key"]
    os.environ["OPENAI_ORGANIZATION"] = args["open_ai_org_key"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=2)
    prompt = PromptTemplate(input_variables=["caption", "table"], template=cot_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # ============= Calculate performance on test set ================= #
    llm_preds = []
    for caption, table in tqdm(zip(double_check_captions, double_check_markdown)):
        answer = chain.run(caption=caption, table=table)
        pred_class = process_llm_output(answer, class_labels)
        pred_label = label2id[pred_class]
        llm_preds.append(pred_label)

    llm_updated_classifier_preds = replace_values_at_indices(preds.tolist(), llm_preds,
                                                  below_threshold_indices.tolist())

    print_model_scores(y_preds=llm_updated_classifier_preds, y_labels=y_test,
                       id2label=id2label, condition_name=f"Combined ZS and Classifier Approach, threshold:{confidence_threshold}")

if __name__ == '__main__':
    typer.run(main)