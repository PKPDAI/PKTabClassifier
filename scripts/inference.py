import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import typer
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from tqdm import tqdm

from pk_tableclass.postprocessing import process_llm_output, replace_values_at_indices
from pk_tableclass.preprocessing import serialize_features, prepare_table_features_inference
from pk_tableclass.prompt_templates import cot_template
from pk_tableclass.utils import make_confidence_histogram


def main(
        path_to_config: Path =typer.Option("configs/config.json",
                                    help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        path_to_trained_model: Path = typer.Option(default="trained_models/best_classifier.pkl", help="Path to trained model, expects .pkl file."),
        inference_data_path: Path = typer.Option("data/inference_data.parquet", help="Path to inference data file, expects .parquet file."),
        confidence_threshold: float = typer.Option(default=0.9, help="Confidence threshold."),
        batch_size: int = typer.Option(default=500, help="Batch size for XGB model predictions."),
):
    # ============= Read in config and get args ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    class_labels = args["class_labels"]
    label2id = {label: idx for idx, label in enumerate(class_labels)}
    id2label = {idx: label for idx, label in enumerate(class_labels)}
    feature_names = args["feature"]

    # ============= Set device ================= #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    print("===========================\n")

    # ============= Load Models ================= #
    # xgb
    pipeline_trained = pickle.load(open(path_to_trained_model, 'rb'))
    # llm
    os.environ['OPENAI_API_KEY'] = args["open_ai_api_key"]
    os.environ["OPENAI_ORGANIZATION"] = args["open_ai_org_key"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=2)
    prompt = PromptTemplate(input_variables=["caption", "table"], template=cot_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # ============= Read in and prepare data ================= #
    print("Prepping data for inference")
    print("===========================\n")
    raw_inf_df = pd.read_parquet(inference_data_path)
    inf_df = prepare_table_features_inference(raw_inf_df)

    # ============= Predict in Batches ================= #
    n_batches = len(inf_df) // batch_size
    preds_inf = []
    preds_inf_probs = []

    for start in tqdm(range(0, len(inf_df), batch_size), total=n_batches, desc="Processing Batches"):
        batch = inf_df.iloc[start:start + batch_size]
        X_batch = serialize_features(feature_names, batch)
        preds_batch = pipeline_trained.predict(X_batch)
        pred_inf_batch_prob = pipeline_trained.predict_proba(X_batch)
        pk_probs = pred_inf_batch_prob.max(axis=1)
        labels_batch = [id2label[x] for x in preds_batch]
        preds_inf_probs.append(pk_probs)
        preds_inf.append(labels_batch)

    preds_inf_probs = np.concatenate(preds_inf_probs)
    preds_inf = np.concatenate(preds_inf)
    make_confidence_histogram(y_confidences=preds_inf_probs, condition_name="Inference Dataset")

    # ============= Calculate performance ================= #
    below_threshold_indices = np.where(preds_inf_probs < confidence_threshold)[0]
    double_check_df = inf_df.iloc[below_threshold_indices]
    print(f"Threshold: {confidence_threshold}, Number check: {len(double_check_df)}.\n")
    double_check_markdown = double_check_df["markdown"].to_list()
    double_check_captions = double_check_df["caption"].to_list()

    llm_preds = []
    for caption, table in tqdm(zip(double_check_captions, double_check_markdown)):
        answer = chain.run(caption=caption, table=table)
        pred_class = process_llm_output(answer, class_labels)
        llm_preds.append(pred_class)

    llm_updated_classifier_preds = replace_values_at_indices(preds_inf.tolist(), llm_preds,
                                                             below_threshold_indices.tolist())

    inf_df["table_relevant"] = llm_updated_classifier_preds
    inf_df["table_relevant_prob"] = preds_inf_probs.tolist()
    print(f"Class Value Counts: {inf_df['table_relevant'].value_counts()}")

    print("========== INFERENCE COMPLETE =============")


if __name__ == "__main__":
    typer.run(main)

