from configparser import ConfigParser
from chain_of_table_pack.llama_index.packs.tables.chain_of_table.base import ChainOfTableQueryEngine
from llama_index.llms.openai import OpenAI
import pandas as pd

import os
import re
from argparse import ArgumentParser

import pandas as pd
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM


config = ConfigParser()
config.read('config.ini')
openai_api_key = config.get('OPENAI_KEY', 'openai-api-key')

llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0)

def create_prompt(question):
    instruction = ""
    prompt = f"{instruction} {question}"
    return prompt

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gri-qa_extra.csv')
    parser.add_argument('type', type=str, default='one-table', choices=['one-table', 'multi-table'])
    args = parser.parse_args()

    qa = pd.read_csv(f'dataset/{args.type}/{args.dataset}', sep=',')
    qa = qa[qa.iloc[:, 2] != 2.0]
    dataset_name = re.split("[_.]", args.dataset)[1]

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
        prompt = create_prompt(row["question"])

        query_engine = ChainOfTableQueryEngine(table, llm=llm, verbose=True)
        response = str(query_engine.query(prompt))
        response_value = response.split('assistant: ')[1]

        print(f'Q{i}: {row["value"]} - {response_value}')

        results.loc[len(results)] = {'index': i, 'question': row["question"], 'value': row["value"], 'response': response_value}

    tracker.stop()

    os.makedirs(f'./results/{args.type}/{dataset_name}', exist_ok=True)
    results.to_csv(f'./results/{args.type}/{dataset_name}/chainoftable.csv', index=False)
    os.rename(f'./results/{args.type}/{dataset_name}/emissions.csv',f'./results/{args.type}/{dataset_name}/emissions_chainoftable.csv')