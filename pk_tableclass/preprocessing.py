from typing import Dict, Any, List, Union

import pandas as pd
from bs4 import BeautifulSoup
from datasets import tqdm
from tqdm import tqdm
from transformers import AutoTokenizer


def html_table_to_markdown_features(html_table: str) -> Dict[str, str]:
    """
    Convert an HTML table to markdown format and extract table features.

    Args:
        html_table (str): A string representing the HTML content of a table.

    Returns:
        Dict[str, str]: A dictionary containing extracted table features:
                        - 'header_row': Markdown representation of the header row.
                        - 'first_few_rows': Markdown representation of the first few rows (up to 3 rows).
                        - 'first_column': Text from the first column concatenated.
                        - 'markdown': Markdown representation of the full table.
    """
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(html_table, "html.parser")

    # Find the first <table> element in the parsed HTML
    table = soup.find("table")

    # Find all rows within the table
    rows: List = table.find_all("tr")

    # Extract header cells (either <th> or <td>) from the first row
    headers = rows[0].find_all("th") or rows[0].find_all("td")
    header_cells: List[str] = [header.get_text(strip=True) for header in headers]

    # Create the markdown representation of the header row
    header_row: str = "| " + " | ".join(header_cells) + " |"

    # Initialize list to store data rows
    data_rows: List[List[str]] = []

    # Extract content from each data row and store the text from each <td> cell
    for row in rows[1:]:
        cells = row.find_all("td")
        data_rows.append([cell.get_text(strip=True) for cell in cells])

    # Initialize markdown table and first few rows (for previewing)
    markdown_table: List[str] = []
    first_few_rows: List[str] = []

    # Add the header row and the alignment row to both markdown tables
    markdown_table.append(header_row)
    first_few_rows.append(header_row)
    markdown_table.append("|" + " --- |" * len(header_cells))  # Alignment row for markdown

    # Add each data row to the markdown table, and to the preview if it's within the first 3 rows
    for i, data_row in enumerate(data_rows):
        row_str: str = "| " + " | ".join(data_row) + " |"
        markdown_table.append(row_str)
        if i < 3:  # Capture only the first three rows for the preview
            first_few_rows.append(row_str)

    # Extract the first column (used for additional features)
    first_column: str = " ".join([row.split(" | ")[0] for row in markdown_table])

    # Join the markdown table and first few rows into a single string
    markdown_table_str: str = "\n".join(markdown_table)
    first_few_rows_str: str = "\n".join(first_few_rows)

    # Create the final dictionary of markdown features
    markdown_features: Dict[str, str] = {
        "header_row": header_row,
        "first_few_rows": first_few_rows_str,
        "first_column": first_column,
        "markdown": markdown_table_str
    }

    return markdown_features


def prepare_table_features(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process each sample from the input DataFrame by converting HTML table content
    to markdown features and creating a new DataFrame with extracted features.

    Args:
        data_df (pd.DataFrame): Input DataFrame containing table data, with columns like
                                'html', 'table_id', 'pmc_link', 'label', 'caption', 'footer'.

    Returns:
        pd.DataFrame: Processed DataFrame with additional features extracted from HTML content
                      (header_row, first_few_rows, first_column, markdown).
    """
    print("======= Preprocessing ========")
    all_processed_samples: List[Dict[str, Any]] = []  # Initialize a list to hold processed sample dictionaries

    # Iterate over each sample in the DataFrame, converting each row to a dictionary
    for sample in tqdm(data_df.to_dict(orient="records")):
        # Parse the HTML table content
        html_table: str = sample["html"]

        # Extract markdown features from the HTML content
        markdown_features: Dict[str, Any] = html_table_to_markdown_features(html_table)

        # Create a dictionary for the processed sample with the relevant fields and markdown features
        processed_sample: Dict[str, Any] = {
            "table_id": sample["table_id"],
            "pmc_link": sample["pmc_link"],
            "label": sample["label"],
            "caption": sample["caption"],
            "html": html_table,
            "footer": sample["footer"],
            "header_row": markdown_features["header_row"],
            "first_few_rows": markdown_features["first_few_rows"],
            "first_column": markdown_features["first_column"],
            "markdown": markdown_features["markdown"]
        }

        # Append the processed sample to the list of all processed samples
        all_processed_samples.append(processed_sample)

    # Convert the list of processed samples back to a DataFrame
    prepared_df: pd.DataFrame = pd.DataFrame(all_processed_samples)

    return prepared_df


def serialize_features(feature_list: List[str], data_df: pd.DataFrame) -> List[str]:
    """
    Serialize features from a DataFrame by combining values from specified columns into a single string.

    Args:
        feature_list (List[str]): List of column names to serialize.
        data_df (pd.DataFrame): Input DataFrame containing the features to be serialized.

    Returns:
        List[str]: A list of serialized feature strings, where each string is a concatenation of
                   values from the specified feature columns for each row.
    """
    ready_list = []  # Initialize a list to hold the final serialized strings

    # Create a list of lists, where each sublist is the column data for the corresponding feature
    input_lists = [data_df[col] for col in feature_list]

    # Iterate over the zipped column values, row by row
    for features in zip(*input_lists):
        nearly_ready_list: List[str] = []  # To hold the serialized parts for the current row

        # Process each feature value in the current row
        for f in features:
            if isinstance(f, list):
                # If the feature value is a list, join the list elements into a single string
                f_string: str = " ".join(f)
                nearly_ready_list.append(f_string)
            else:
                # Otherwise, just append the value as it is
                nearly_ready_list.append(str(f))  # Convert to string to handle non-list types

        # Join the serialized values for the current row into a single string
        nearly_ready_string: str = " ".join(nearly_ready_list)
        ready_list.append(nearly_ready_string)

    return ready_list


def chunk_texts(texts: List[str], pretrained_model_path: str) -> List[List[str]]:
    """
    Chunk a list of input texts into smaller text segments based on the tokenization of a pretrained model.
    Each text is split into chunks that fit within the model's maximum token length.

    Args:
        texts (List[str]): List of input texts to be chunked.
        pretrained_model_path (str): Path to the pretrained model for tokenization.

    Returns:
        List[List[str]]: A list of lists, where each sublist contains chunks of the corresponding input text.
                         Each chunk is a string that fits within the token limit of the model.
    """
    # Load the tokenizer from the pretrained model path
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

    # Define the maximum length for each chunk based on the model's token limit
    max_length: int = tokenizer.model_max_length - 2  # Reserve space for special tokens
    min_chunk_length: int = 1  # Minimum token length required for a valid chunk

    # Tokenize the input texts without truncating and without adding special tokens
    tokenized_texts: List[List[int]] = tokenizer(texts, truncation=False, add_special_tokens=False)['input_ids']

    all_chunks: List[List[str]] = []  # To store the chunked texts

    # Iterate over each tokenized text
    for tokens in tokenized_texts:
        # Break the token sequence into chunks of max_length
        chunks: List[List[int]] = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
        example_chunks: List[str] = []  # To store the decoded chunks for this example

        # Decode each chunk back to text and filter out chunks below the minimum length
        for chunk in chunks:
            if len(chunk) >= min_chunk_length:
                chunk_text: str = tokenizer.decode(chunk, skip_special_tokens=True)
                example_chunks.append(chunk_text)

        # Append the chunks of the current text to the all_chunks list
        all_chunks.append(example_chunks)

    # Ensure that the number of chunked texts matches the number of input texts
    assert len(all_chunks) == len(texts), "The number of chunked texts should match the input texts."

    return all_chunks


def flatten_to_strings(data: List[Union[str, List[str]]]) -> List[str]:
    """
    Flatten a list of lists into a single list of strings.

    Args:
        data (List[Union[str, List[str]]]): A list that contains both strings and lists of strings.

    Returns:
        List[str]: A flattened list where all nested lists are expanded into individual strings.
    """
    flattened_list: List[str] = []  # Initialize an empty list to store the flattened results

    # Iterate over each item in the input data
    for item in data:
        if isinstance(item, list):
            # If the item is a list, extend the result list with the elements of that list
            flattened_list.extend(item)
        else:
            # If the item is not a list (assumed to be a string), append it directly to the result list
            flattened_list.append(item)

    return flattened_list


def combine_features(
        feature_list: List[str],
        data_df: pd.DataFrame,
        pretrained_model_path: str,
        chunking: bool
) -> List[Union[str, List[str]]]:
    """
    Combine multiple features from a DataFrame into a serialized form, optionally applying chunking
    based on a pretrained model. If features are lists, they will be flattened or serialized before combination.

    Args:
        feature_list (List[str]): List of column names to be combined.
        data_df (pd.DataFrame): Input DataFrame containing the features.
        pretrained_model_path (str): Path to a pretrained model used for text chunking, if applicable.
        chunking (bool): Boolean flag indicating whether to apply chunking to the features.

    Returns:
        List[Union[str, List[str]]]: A list where each element is a combined feature string or list of strings
                                     for each row in the DataFrame.
    """
    input_lists: List[List[Union[str, List[str]]]] = []  # To store the feature columns' content

    # Iterate over the feature names to process each feature
    for f in feature_list:
        feature = data_df[f].to_list()  # Convert the DataFrame column to a list

        # Check if the feature is a list and handle chunking if required
        if isinstance(feature[0], list):
            if chunking:
                # Serialize features and apply chunking based on the pretrained model
                feature = serialize_features([f], data_df=data_df)
                feature = chunk_texts(feature, pretrained_model_path=pretrained_model_path)
            input_lists.append(feature)
        else:
            # Directly append the feature list if not a list of lists
            input_lists.append(feature)

    ready_features: List[List[str]] = []  # To store the flattened or serialized features

    # Combine features for each row by zipping the feature columns
    for features in zip(*input_lists):
        # Flatten or combine features into a single string
        ready_feature: List[str] = flatten_to_strings(features)
        ready_features.append(ready_feature)

    # If any list contains only a single element, simplify it to that element
    flattened_list: List[Union[str, List[str]]] = [
        sublist[0] if len(sublist) == 1 else sublist for sublist in ready_features
    ]

    return flattened_list


def prepare_table_features_inference(data_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process each sample from the input DataFrame by converting HTML table content
    to markdown features and creating a new DataFrame with extracted features.

    Args:
        data_df (pd.DataFrame): Input DataFrame containing table data, with columns like
                                'html', 'table_id', 'pmc_link', 'label', 'caption', 'footer'.

    Returns:
        pd.DataFrame: Processed DataFrame with additional features extracted from HTML content
                      (header_row, first_few_rows, first_column, markdown).
    """
    print("======= Preprocessing ========")
    all_processed_samples: List[Dict[str, Any]] = []

    for sample in tqdm(data_df.to_dict(orient="records")):
        # Parse the HTML table content
        html_table: str = sample["html_table"]
        markdown_features: Dict[str, Any] = html_table_to_markdown_features(html_table)

        if "label" in sample:
            label = sample["label"]
        else:
            label = ""

        processed_sample: Dict[str, Any] = {
            "pmc_link": sample["pmc_link"],
            "pmid": sample["pmid"],
            "pmc": sample["pmc"],
            "table_number": sample["table_number"],
            "caption": sample["caption"],
            "html": html_table,
            "markdown": markdown_features["markdown"],
            "label": label,
        }

        all_processed_samples.append(processed_sample)

    prepared_df: pd.DataFrame = pd.DataFrame(all_processed_samples)

    return prepared_df



def reformat_tables_data(data):
    """
    Reformat the list of dictionaries to flatten out tables.

    Parameters:
        data (list): A list of dictionaries, each containing article info and a list of tables.

    Returns:
        list: A list of reformatted dictionaries, where each table is its own entry.
    """
    reformatted_data = []

    print("Reformatting tables...\n\n")
    for article in tqdm(data):
        pmc_link = article.get("pmc_link")
        pmid = article.get("pmid")
        pmc = article.get("pmc")
        article_title = article.get("article_title")

        for table in article.get("tables", []):
            table_entry = {
                "pmc_link": pmc_link,
                "pmid": pmid,
                "pmc": pmc,
                "article_title": article_title,
                "table_id": table.get("table_id"),
                "table_number": table.get("table_number"),
                "html_table": table.get("html_table"),
                "caption": table.get("caption"),
                "footer": table.get("footer"),
            }
            reformatted_data.append(table_entry)
        a = 1
    a = 1
    return reformatted_data