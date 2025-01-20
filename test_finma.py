import os
import random
import re
from argparse import ArgumentParser
from configparser import ConfigParser

import pandas as pd
from codecarbon import EmissionsTracker
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


def flattening(table):
    flatten_table = ""

    for i, line in table.iterrows():
        tab_line = ""
        for cell in line:
            tab_line += f" {cell},"
        flatten_table += (tab_line + "\n")

    return flatten_table


def create_prompt(row, type):
    table_dirnames = eval(row["pdf name"])

    tables = []
    for i, table_dirname in enumerate(table_dirnames):
        table_dirname = table_dirname.split(".")[0]
        table_filename = f'dataset/annotation/{table_dirname}/{eval(row["page nbr"])[i]}_{eval(row["table nbr"])[i]}.csv'
        table = pd.read_csv(table_filename, sep=';')
        if type == 'one-table':
            tables.append(flattening(table))
        else:
            company = table_dirname.strip('_2023')
            table = f"Company name: {company}\n\n{flattening(table)}"
            tables.append(table)

    tables = "\n\n".join(tables)
    instruction = "Given the financial data and expert analysis, please answer this question:"

    return f'{instruction}\nContext: {tables}\nQuestion: {row["question"]}\nResponse:'


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gri-qa_multitable2.csv')
    parser.add_argument('--type', type=str, default='multi-table', choices=['one-table', 'multi-table'])
    args = parser.parse_args()

    set_seed(42)
    random.seed(42)

    qa = pd.read_csv(f'dataset/{args.type}/{args.dataset}', sep=',')
    dataset_name = re.split("[_.]", args.dataset)[1]

    config = ConfigParser()
    config.read('config.ini')
    token = config.get('TOKEN', 'token-hf')

    login(token=token)

    tokenizer = AutoTokenizer.from_pretrained('TheFinAI/finma-7b-full')
    model = AutoModelForCausalLM.from_pretrained(
        'TheFinAI/finma-7b-full',
        device_map='auto'
    )

    results = pd.DataFrame(columns=['index', 'question', 'value', 'response'])

    os.makedirs(f'./results/{args.type}/{dataset_name}', exist_ok=True)
    tracker = EmissionsTracker(output_dir=f'./results/{args.type}/{dataset_name}')

    tracker.start()

    for i, row in qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        # Query
        prompt = create_prompt(row, args.type)

        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**encoding, max_length=2000)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_value = response[0].strip('[\'').split('\nResponse: ')[1].strip('\']')
        print(f'Q{i}: {row["value"]} - {response_value}')

        results.loc[len(results)] = {'index': i, 'question': row["question"], 'value': row["value"], 'response': response_value}

    tracker.stop()

    results.to_csv(f'./results/{args.type}/{dataset_name}/finma.csv', index=False)
    os.rename(f'./results/{args.type}/{dataset_name}/emissions.csv',f'./results/{args.type}/{dataset_name}/emissions_finma.csv')
