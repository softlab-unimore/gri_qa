import os
import random
import re
from argparse import ArgumentParser

import pandas as pd
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


def flattening(table):
    flatten_table = ""

    header = "[TAB] " + " | ".join(table.columns) + " [SEP]\n"
    flatten_table += header

    for i, line in table.iterrows():
        tab_line = ""
        for cell in line:
            tab_line += f" {cell} |"
        tab_line += " [SEP]"
        flatten_table += tab_line + "\n"

    return flatten_table


def create_prompt(row, hierarchical, args):
    table_dirnames = eval(row["pdf name"])

    tables = []
    for i, table_dirname in enumerate(table_dirnames):
        table_dirname = table_dirname.split(".")[0]
        table_filename = f'dataset/annotation/{table_dirname}/{eval(row["page nbr"])[i]}_{eval(row["table nbr"])[i]}.csv'
        table = pd.read_csv(table_filename, sep=';')
        if args.type == 'one-table' and 'multitable' not in args.dataset:
            tables.append(flattening(table))
        else:
            company = table_dirname.strip('_2023')
            table = f"Company name: {company}\n\n{flattening(table)}"
            tables.append(table)

    description = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

    if not hierarchical:
        if args.type == 'one-table' and 'multitable' not in args.dataset:
            instruction = f"This is a table QA task. The goal of this task is to answer the question given the table."
        else:
            instruction = f'This is a table QA task. The goal of this task is to answer the question given some tables.'
    else:
        if args.type == 'one-table' and 'multitable' not in args.dataset:
            instruction = f"This is a hierarchical table QA task. The goal of this task is to answer the question given the table. The table might be hierarchical."
        else:
            instruction = f'This is a hierarchical table QA task. The goal of this task is to answer the question given some tables. The tables might be hierarchical.'

    tables = "\n\n".join(tables)
    prompt = f"{description}\n\n### Instruction:\n{instruction}\n\n### Input:\n{tables}\n\n### Question:\n{row['question']}\n\n### Response:\n"
    return prompt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='samples-gri-qa_multitable2-multistep.csv')
    parser.add_argument('--type', type=str, default='samples', choices=['one-table', 'samples', 'multi-table'])
    args = parser.parse_args()

    set_seed(42)
    random.seed(42)

    qa = pd.read_csv(f'dataset/{args.type}/{args.dataset}', sep=',')
    dataset_name = re.split("[_.]", args.dataset)[1]

    tokenizer = AutoTokenizer.from_pretrained('osunlp/TableLlama')
    model = AutoModelForCausalLM.from_pretrained(
        'osunlp/TableLlama',
        device_map='auto'
    )

    results = pd.DataFrame(columns=['index', 'question', 'value', 'response'])

    os.makedirs(f'./results/{args.type}/{dataset_name}', exist_ok=True)
    tracker = EmissionsTracker(output_dir=f'./results/{args.type}/{dataset_name}')

    tracker.start()

    for i, row in qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        try:
            hierarchical = row['hierarchical'] == 1
        except KeyError:
            hierarchical = False

        prompt = create_prompt(row, hierarchical, args)

        # Query
        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        if encoding['input_ids'].shape[1] < 4096:
            outputs = model.generate(**encoding)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            response_value = response[0].split('### Response:\n')[1]

            print(f'Q{i}: {row["value"]} - {response_value}')

            results.loc[len(results)] = {'index': i, 'question': row["question"], 'value': row["value"], 'response': response_value}
        else:
            print(f'Q{i}: {row["value"]} - Prompt too long')
            results.loc[len(results)] = {'index': i, 'question': row["question"], 'value': row["value"], 'response': 'Prompt too long'}

    tracker.stop()

    results.to_csv(f'./results/{args.type}/{dataset_name}/tablellama.csv', index=False)
    os.rename(f'./results/{args.type}/{dataset_name}/emissions.csv',f'./results/{args.type}/{dataset_name}/emissions_tablellama.csv')
