import os
import re
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import pandas as pd
from openai import OpenAI
from codecarbon import EmissionsTracker


def read_csv_with_encoding(file_path, try_encodings=None):
    """
    Try to read a CSV file with different encodings.

    Args:
        file_path (str): Path to the CSV file
        try_encodings (list): List of encodings to try. If None, uses default list

    Returns:
        pandas.DataFrame: The loaded DataFrame
    """
    if try_encodings is None:
        try_encodings = [
            'utf-8',
            'utf-8-sig',  # UTF-8 with BOM
            'iso-8859-1',
            'cp1252',
            'latin1'
        ]

    for encoding in try_encodings:
        try:
            # Try reading with current encoding
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=';',
                header=None,
                # Additional parameters you might need:
                # sep=',',  # or ';' or '\t' depending on your file
                # low_memory=False,  # for large files
                # on_bad_lines='warn'  # or 'skip' to handle problematic lines
            )
            # print(f"Successfully read file with {encoding} encoding")
            return df

        except UnicodeDecodeError:
            # print(f"Failed with {encoding} encoding, trying next...")
            continue

    raise ValueError("Could not read the file with any of the attempted encodings")


# Table Meets LLM: Can Large Language Models Understand Structured Table Data? A Benchmark and Empirical Study
def df_to_html_string(df: pd.DataFrame, index: bool = True) -> str:
    """Converts DataFrame to styled HTML string.

    Args:
        df (pd.DataFrame): DataFrame to convert
        index (bool): Whether to include index in HTML output

    Returns:
        str: HTML string representation of the DataFrame
    """
    
    return df.to_html(index=index, classes='table table-striped table-bordered', border=1, justify='left', escape=False)

def answer(question: str, table: pd.DataFrame, dataset_file: str) -> str:
    """Generates answer for a question about table data using OpenAI's API.

    Args:
        question (str): Question to be answered about the table
        table (pd.DataFrame): Table data to be analyzed
        dataset_file (str): Path to the dataset file

    Returns:
        str: AI-generated answer based on the table content
    """
    # Initialize OpenAI client and generate response based on question and table
    client = OpenAI()
    rel = "If the question asks for a boolean value, the answer should be 'yes' or 'no'."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"{question}\n{df_to_html_string(table, index=False)}\n\n"
                           f"Give exclusively the numerical answer. Do not write anything else. "
                           f"Do not write any markdown formatting. {rel if 'rel' in dataset_file else ''}"
            }
        ],
        temperature=0.0
    )
    response = completion.choices[0].message.content
    return response


def table_predictions(qa_file: str, dataset_dir: str, save_dir: str) -> pd.DataFrame:
    """Processes Q&A pairs by analyzing tables and generating predictions.

    Args:
        qa_file (str): Path to CSV file containing questions and table references
        dataset_dir (str): Directory containing table CSV files organized by PDF
        save_dir(str): Directory to save the results

    Returns:
        pd.DataFrame: Results containing original questions, true values, and predictions
    """
    # Load initial QA data and prepare index
    df_qa = pd.read_csv(qa_file)
    df_qa = df_qa[['question', 'pdf name', 'page nbr', 'table nbr', 'value']].astype(str)
    df_qa['index'] = np.arange(len(df_qa))
    df_qa = df_qa[df_qa.iloc[:, 2] != 2.0]

    # Initialize CodeCarbon tracker
    os.makedirs(save_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=save_dir)
    tracker.start()

    # Process each question against its corresponding table
    predictions = []
    for i, row in df_qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        pdf_name, page_num, table_num = row['pdf name'], row['page nbr'], row['table nbr']
        pdf_name = pdf_name.replace('.pdf', '')
        table_path = os.path.join(dataset_dir, pdf_name, f'{page_num}_{table_num}.csv')

        if not os.path.exists(table_path):
            print(f'{table_path} does not exist')
            predictions.append(None)
            continue

        table = read_csv_with_encoding(str(table_path))
        pred = answer(row['question'], table, qa_file)
        predictions.append(pred)

        print(f'Q{i}: {row["value"]} - {pred}')

    # Create response column
    df_qa['response'] = predictions

    # Stop CodeCarbon tracker
    tracker.stop()

    return df_qa[['index', 'question', 'value', 'response']]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gri-qa_extra.csv')
    args = parser.parse_args()

    config = ConfigParser()
    config.read('config.ini')
    os.environ['OPENAI_API_KEY'] = config.get('TOKEN', 'token_openai')

    dataset_name = re.split("[_.]", args.dataset)[1]

    df_preds = table_predictions(f'dataset/{args.dataset}', './dataset/annotation', f'./results/{dataset_name}')
    df_preds.to_csv(f'./results/{dataset_name}/openai.csv', index=False)
    os.rename(f'./results/{dataset_name}/emissions.csv', f'./results/{dataset_name}/emissions_openai.csv')
