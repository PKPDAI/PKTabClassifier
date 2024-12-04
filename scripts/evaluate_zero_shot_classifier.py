import json
import os
from pathlib import Path

import pandas as pd
import typer
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.metrics import classification_report
from tqdm import tqdm

from pk_tableclass.postprocessing import process_llm_output
from pk_tableclass.preprocessing import prepare_table_features
from pk_tableclass.prompt_templates import cot_template




def main(
        path_to_config: Path =typer.Option("configs/config.json",
                                    help="Path to config file specifying classifier pipeline arguments, expects .json file."),
        test_data_path: Path =typer.Option("data/test.pkl", help="Path to test data file, expects .pkl file.")
):
    # ============= Read in data ================= #
    with open(path_to_config, 'r') as file:
        args = json.load(file)
    class_labels = args["class_labels"]
    id2label = {idx: label for idx, label in enumerate(class_labels)}
    raw_test_data = pd.read_pickle(test_data_path)
    test_df = prepare_table_features(raw_test_data)

    test_markdown = test_df["markdown"].to_list()
    test_captions = test_df["caption"].to_list()
    y_test = test_df["label"].to_list()

    # ============= Define llm, prompt & chain ================= #
    os.environ['OPENAI_API_KEY'] = args["open_ai_api_key"]
    os.environ["OPENAI_ORGANIZATION"] = args["open_ai_org_key"]
    llm = ChatOpenAI(model="gpt-4o", temperature=0, max_retries=2)
    prompt = PromptTemplate(input_variables=["caption", "table"], template=cot_template)
    chain = LLMChain(llm=llm, prompt=prompt)

    # ============= Calculate performance on unseen test set ================= #
    llm_preds = []
    for caption, table in tqdm(zip(test_captions, test_markdown)):
        answer = chain.run(caption=caption, table=table)
        pred_class = process_llm_output(answer, class_labels)
        llm_preds.append(pred_class)

    zs_class_report = classification_report(llm_preds, y_test, output_dict=True, target_names=[id2label[i] for i in range(len(id2label))])
    df = pd.DataFrame(zs_class_report).transpose()
    metrics_columns = ['precision', 'recall', 'f1-score']
    df[metrics_columns] = df[metrics_columns] * 100
    df_rounded = df.round(2)
    print(df_rounded)


if __name__ == '__main__':
    typer.run(main)