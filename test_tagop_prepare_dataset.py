import os
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer

from tag_op.data.tatqa_dataset import TagTaTQATestReader


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


def save_dict_to_json(dictionary: list, file_path, indent=4, encoding='utf-8'):
    """
    Save a dictionary as a JSON file.

    Args:
        dictionary (dict): The dictionary to save
        file_path (str): Path where to save the JSON file
        indent (int): Number of spaces for indentation
        encoding (str): File encoding
    """
    try:
        # Convert to Path object
        path = Path(file_path)

        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write the JSON file
        with open(path, 'w', encoding=encoding) as f:
            json.dump(dictionary, f, indent=indent, ensure_ascii=False)

        print(f"Successfully saved JSON file to: {path}")

    except Exception as e:
        print(f"Error saving JSON file: {str(e)}")


def qa_dataset_to_tagop(qa_file: str, dataset_dir: str) -> list:
    """Converts a QA dataset CSV file into TagOp format for table question answering.

    Processes questions about tables from a CSV file and matches them with their
    corresponding table files to create a TagOp-compatible dataset structure.

    Args:
        qa_file: Path to the CSV file containing questions and table references.
        dataset_dir: Directory containing the table files organized by PDF.

    Returns:
        List of dictionaries containing tables and their associated questions in
        TagOp format.
    """
    df = pd.read_csv(qa_file)
    df = df[['question', 'pdf name', 'page nbr', 'table nbr', 'value']].astype(str)
    df['idx'] = np.arange(len(df))

    tagop_dataset = []
    # Group questions by their associated table
    tables_iter = dict(list(df.groupby(['pdf name', 'page nbr', 'table nbr'])))
    for key, questions in tables_iter.items():
        pdf_name, page_num, table_num = key
        pdf_name = pdf_name.replace('.pdf', '')
        table_path = os.path.join(dataset_dir, pdf_name, f'{page_num}_{table_num}.csv')

        if not os.path.exists(table_path):
            print(f'{table_path} does not exist')
            continue

        # Load and prepare table data
        table = read_csv_with_encoding(str(table_path))
        table = table.fillna('')
        table = table.astype(str)
        # Transpose table for TagOp format requirement
        table_list = table.T.values.tolist()
        table_id = f'{pdf_name}__{page_num}_{table_num}'

        # Format questions for current table
        questions_list = []
        for i in range(len(questions)):
            idx = questions['idx'].iloc[i]
            question = questions['question'].iloc[i]
            questions_list.append({
                'uid': str(idx),
                'order': i + 1,
                'question': question,
            })

        tagop_dataset.append({
            'table': {'uid': table_id, 'table': table_list},
            'questions': questions_list,
            'paragraphs': []
        })

    return tagop_dataset


def tagop_prepare_dataset(input_path: str, roberta_model: str, output_path: str):
    """Prepare a TagOp dataset for prediction with a specific encoder.

    Args:
        input_path: Path to the JSON file containing the TagOp dataset.
        roberta_model: Path to the Roberta model for tokenization.
        output_path: Path to save the prepared dataset in pickle format.
    Returns:
        None
    """
    passage_length_limit = 463
    question_length_limit = 46
    encoder = 'roberta'

    if encoder == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained(roberta_model)
        sep = '<s>'
    elif encoder == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        sep = '[SEP]'
    else:
        raise ValueError(f"Invalid encoder: {encoder}")

    data_reader = TagTaTQATestReader(tokenizer, passage_length_limit, question_length_limit, sep=sep)

    print(f'==== NOTE ====: encoder:{encoder}, mode:dev')

    data = data_reader._read(input_path)
    print(data_reader.skip_count)
    data_reader.skip_count = 0
    print(f"Save data to {output_path}")

    with open(output_path, "wb") as f:
        pickle.dump(data, f)


# We adopt `RoBERTa` as our encoder to develop our TagOp and use the following commands to prepare RoBERTa model
#
# ```bash
# cd dataset_tagop
# mkdir roberta.large && cd roberta.large
# wget -O pytorch_model.bin https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin
# wget -O config.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json
# wget -O vocab.json https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json
# wget -O merges.txt https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-merges.txt
# ```

def main():
    qa_file = './dataset/qa_dataset.csv'
    dataset_dir = './dataset/annotation'
    tagop_data_path = './dataset_tagop'
    roberta_model = './dataset_tagop/roberta.large' # ToDo: download the model

    # Step 1: Convert QA dataset to TagOp format
    tagop_dataset = qa_dataset_to_tagop(qa_file, dataset_dir)

    # Step 2: Save TagOp dataset to JSON file
    out_json_file = os.path.join(tagop_data_path, 'qa_tagop_dataset.json')
    save_dict_to_json(tagop_dataset, out_json_file)

    # Step 3: Prepare TagOp dataset for prediction
    out_pkl_file = os.path.join(tagop_data_path, 'cache', 'qa_tagop_dataset.pkl')
    os.makedirs(os.path.join(tagop_data_path, 'cache'), exist_ok=True)
    tagop_prepare_dataset(out_json_file, roberta_model, out_pkl_file)

    print('Finished')


if __name__ == '__main__':
    main()
