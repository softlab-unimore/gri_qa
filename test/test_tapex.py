import os
import random
import re
from argparse import ArgumentParser

import pandas as pd
from codecarbon import EmissionsTracker
from transformers import TapexTokenizer, BartForConditionalGeneration, set_seed

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gri-qa_extra.csv')
    parser.add_argument('--type', type=str, default='one-table', choices=['one-table', 'multi-table', 'samples'])
    args = parser.parse_args()

    set_seed(42)
    random.seed(42)

    qa = pd.read_csv(f'dataset/{args.type}/{args.dataset}', sep=',')
    dataset_name = re.split("[_.]", args.dataset)[1]

    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    model = BartForConditionalGeneration.from_pretrained(
        "microsoft/tapex-large-finetuned-wtq",
        device_map='auto',
    )

    results = pd.DataFrame(columns=['index', 'question', 'value', 'response'])

    os.makedirs(f'./results/{args.type}/{dataset_name}', exist_ok=True)
    tracker = EmissionsTracker(output_dir=f'./results/{args.type}/{dataset_name}')

    tracker.start()

    for i, row in qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        # Table extraction
        table_dirnames = eval(row["pdf name"])[0].split('.')
        tables = []
        for table_dirname in table_dirnames:
            table_filename = f'dataset/annotation/{table_dirname}/{str(eval(row["page nbr"])[0]).split(".")[0]}_{str(eval(row["table nbr"])[0]).split(".")[0]}.csv'
            table = pd.read_csv(table_filename, sep=';')
            table = table.astype(str)
            tables.append(table)

        # Query
        encoding = tokenizer(table=tables, query=row["question"], return_tensors="pt").to(model.device)
        if encoding['input_ids'].shape[1] < 1024:
            outputs = model.generate(**encoding)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            print(f'Q{i}: {row["value"]} - {response[0]}')

            results.loc[len(results)] = {'index': i, 'question': row["question"], 'value': row["value"], 'response': response[0]}
        else:
            print(f'Q{i}: {row["value"]} - Table too long')

            results.loc[len(results)] = {'index': i, 'question': row["question"], 'value': row["value"], 'response': 'Table too long'}

    tracker.stop()

    results.to_csv(f'./results/{args.type}/{dataset_name}/tapex.csv', index=False)
    os.rename(f'./results/{args.type}/{dataset_name}/emissions.csv',f'./results/{args.type}/{dataset_name}/emissions_tapex.csv')
