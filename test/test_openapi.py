import os
import re
from argparse import ArgumentParser
from configparser import ConfigParser

import numpy as np
import pandas as pd
from openai import OpenAI
from codecarbon import EmissionsTracker

from phoenix.otel import register
from openinference.instrumentation.openai import OpenAIInstrumentor

tracer_provider = register(
    project_name="openai",
    endpoint="http://localhost:6006/v1/traces",
)

OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)

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
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                sep=';',
                header=None,
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

def answer(question: str, tables: str, dataset_file: str, type: str) -> str:
    """Generates answer for a question about table data using OpenAI's API.

    Args:
        question (str): Question to be answered about the table
        table (pd.str): Table data to be analyzed concatenated as a string
        dataset_file (str): Path to the dataset file
        type (str): Type of table data (one-table or multi-table)

    Returns:
        str: AI-generated answer based on the table content
    """
    # Initialize OpenAI client and generate response based on question and table
    client = OpenAI()
    instruction = f"You must answer the following question given the provided table{'s' if type == 'multi-table' or 'multitable' in args.dataset else ''}. If the question is boolean, write exclusively a 'yes' or 'no' answer. If the question asks for a list of values, separate them with a comma. Write the numerical values with exactly 2 decimal values. Do not write anything else. Do not write any Markdown formatting."
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"{instruction}\n\nQuestion: {question}\nTable{'s' if type == 'multi-table' or 'multitable' in args.dataset else ''}: {tables}\n\n"
            }
        ],
        temperature=0.0
    )
    response = completion.choices[0].message.content.lower().split('the final answer is:')[-1]
    return response


def table_predictions(qa_file: str, dataset_dir: str, save_dir: str, type: str) -> pd.DataFrame:
    """Processes Q&A pairs by analyzing tables and generating predictions.

    Args:
        qa_file (str): Path to CSV file containing questions and table references
        dataset_dir (str): Directory containing table CSV files organized by PDF
        save_dir(str): Directory to save the results
        type (str): Type of table data (one-table or multi-table)

    Returns:
        pd.DataFrame: Results containing original questions, true values, and predictions
    """
    # Load initial QA data and prepare index
    df_qa = pd.read_csv(qa_file)
    df_qa['index'] = df_qa.index
    df_qa = df_qa[['index', 'question', 'pdf name', 'page nbr', 'table nbr', 'value' ]].astype(str)
    df_qa = df_qa.reset_index(drop=True)

    # Initialize CodeCarbon tracker
    os.makedirs(save_dir, exist_ok=True)
    tracker = EmissionsTracker(output_dir=save_dir)
    tracker.start()

    # Process each question against its corresponding table
    predictions = []
    for i, row in df_qa.iterrows():

        print(f'Q{row["index"]} - {row["question"]}')

        pdf_names, page_nums, table_nums = eval(row['pdf name']), eval(row['page nbr']), eval(row['table nbr'])

        tables = []
        for pdf_name, page_num, table_num in zip(pdf_names, page_nums, table_nums):
            pdf_name = pdf_name.replace('.pdf', '')
            table_path = os.path.join(dataset_dir, pdf_name, f'{page_num}_{table_num}.csv')

            if not os.path.exists(table_path):
                print(f'{table_path} does not exist')
                predictions.append(None)
                continue

            company = pdf_name.strip('_2023')
            table = read_csv_with_encoding(str(table_path))
            if type == 'multi-table' or 'multitable' in args.dataset:
                table = f"Company name: {company}\n\n{df_to_html_string(table, index=False)}"
                tables.append(table)
            else:
                tables.append(df_to_html_string(table, index=False))

        tables = "\n\n".join(tables)
        pred = answer(row['question'], tables, qa_file, type=type)
        predictions.append(pred)

        print(f'Q{row["index"]}: {row["value"]} - {pred}')

    # Create response column
    df_qa['response'] = predictions

    # Stop CodeCarbon tracker
    tracker.stop()

    return df_qa[['index', 'question', 'value', 'response']]


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gri-qa_extra.csv')
    parser.add_argument('--type', type=str, default='one-table', choices=['one-table', 'multi-table', 'samples'])
    args = parser.parse_args()

    config = ConfigParser()
    config.read('config.ini')
    os.environ['OPENAI_API_KEY'] = config.get('TOKEN', 'token_openai')

    dataset_name = re.split("[_.]", args.dataset)[1]

    df_preds = table_predictions(f'dataset/{args.type}/{args.dataset}', './dataset/annotation', f'./results/{args.type}/{dataset_name}', type=args.type)
    df_preds.to_csv(f'./results/{args.type}/{dataset_name}/openai.csv', index=False)
    os.rename(f'./results/{args.type}/{dataset_name}/emissions.csv', f'./results/{args.type}/{dataset_name}/emissions_openai.csv')
