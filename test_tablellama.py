import os
import re
from argparse import ArgumentParser

import pandas as pd
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def create_prompt(table, question, hierarchical):
    description = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."

    if not hierarchical:
        instruction = "This is a table QA task. The goal of this task is to answer the question given the table."
    else:
        instruction = "This is a hierarchical table QA task. The goal of this task is to answer the question given the table. The table might be hierarchical."

    table = flattening(table)

    prompt = f"{description}\n\n###Instruction:\n{instruction}\n\nInput:\n{table}\n\n###Question:\n{question}\n\n###Response:\n"
    return prompt


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gri-qa_extra.csv')
    parser.add_argument('--type', type=str, default='one-table', choices=['one-table', 'multi-table'])
    args = parser.parse_args()

    qa = pd.read_csv(f'dataset/{args.type}/{args.dataset}', sep=',')
    qa = qa[qa.iloc[:, 2] != 2.0]
    dataset_name = re.split("[_.]", args.dataset)[1]

    tokenizer = AutoTokenizer.from_pretrained('osunlp/TableLlama')
    model = AutoModelForCausalLM.from_pretrained(
        'osunlp/TableLlama',
        device_map='auto'
    )

    results = pd.DataFrame(columns=['index', 'question', 'value', 'response'])

    tracker = EmissionsTracker(output_dir=f'./results/{args.type}/{dataset_name}')

    tracker.start()

    for i, row in qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        # Table extraction
        table_dirname = row["pdf name"].split('.')[0]
        table_filename = f'dataset/annotation/{table_dirname}/{row["page nbr"]}_{row["table nbr"]}.csv'
        table = pd.read_csv(table_filename, sep=';', on_bad_lines='skip')

        hierarchical = row.iloc[2] == 3
        prompt = create_prompt(table, row["question"], hierarchical)

        # Query
        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**encoding)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_value = response[0].split('###Response:\n')[1]

        print(f'Q{i}: {row["value"]} - {response_value}')

        results.loc[len(results)] = {'index': i, 'question': row["question"], 'value': row["value"], 'response': response_value}

    tracker.stop()

    os.makedirs(f'./results/{args.type}/{dataset_name}', exist_ok=True)
    results.to_csv(f'./results/{args.type}/{dataset_name}/tablellama.csv', index=False)
    os.rename(f'./results/{args.type}/{dataset_name}/emissions.csv',f'./results/{args.type}/{dataset_name}/emissions_tablellama.csv')
