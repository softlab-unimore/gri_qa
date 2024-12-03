import os
from configparser import ConfigParser

import pandas as pd
from codecarbon import EmissionsTracker
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM


def flattening(table):
    flatten_table = ""

    for i, line in table.iterrows():
        tab_line = ""
        for cell in line:
            tab_line += f" {cell},"
        flatten_table += (tab_line + "\n")

    return flatten_table


def create_prompt(table, question):

    instruction = "Given the financial data and expert analysis, please answer this question:"
    table = flattening(table)
    return f'{instruction}\nContext: {table}\nQuestion: {question}\nAnswer:'


if __name__=='__main__':
    qa = pd.read_csv('dataset/qa_dataset.csv', sep=',', on_bad_lines='skip')

    config = ConfigParser()
    config.read('config.ini')
    token = config.get('TOKEN', 'token')

    login(token=token)

    tokenizer = AutoTokenizer.from_pretrained('TheFinAI/finma-7b-full')
    model = AutoModelForCausalLM.from_pretrained(
        'TheFinAI/finma-7b-full',
        device_map='auto'
    )

    results = pd.DataFrame(columns=['question', 'value', 'response'])

    tracker = EmissionsTracker(output_dir='./results')

    tracker.start()

    for i, row in qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        # Table extraction
        table_dirname = row["pdf name"].split('.')[0]
        table_filename = f'dataset/annotation/{table_dirname}/{row["page nbr"]}_{row["table nbr"]}.csv'
        table = pd.read_csv(table_filename, sep=';', on_bad_lines='skip')

        # Query
        prompt = create_prompt(table, row['question'])

        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**encoding, max_length=2000)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        response_value = response[0].strip('[\'').split('\nAnswer: ')[1].strip('\']')
        print(f'Q{i}: {row["value"]} - {response_value}')

        results.loc[len(results)] = {'question': row["question"], 'value': row["value"], 'response': response_value}

    tracker.stop()

    os.makedirs('./results', exist_ok=True)
    results.to_csv(f'./results/finma.csv', index=False)
    os.rename('./results/emissions.csv', './results/emissions_finma.csv')