import os

import pandas as pd
from transformers import TapexTokenizer, BartForConditionalGeneration

if __name__ == '__main__':
    qa = pd.read_csv('dataset/qa_dataset.csv', skiprows=1, sep=',', on_bad_lines='skip')

    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")

    results = pd.DataFrame(columns=['question', 'value', 'response'])

    for i, row in qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        # Table extraction
        table_dirname = row["pdf name"].split('.')[0]
        table_filename = f'dataset/annotation/{table_dirname}/{row["page nbr"]}_{row["table nbr"]}.csv'
        table = pd.read_csv(table_filename, sep=';', on_bad_lines='skip')
        table = table.astype(str)

        # Query
        encoding = tokenizer(table=table, query=row["question"], return_tensors="pt")
        if encoding['input_ids'].shape[1] < 1024:
            outputs = model.generate(**encoding)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            print(f'Q{i}: {row["value"]} - {response[0]}')

            results.loc[len(results)] = {'question': row["question"], 'value': row["value"], 'response': response[0]}
        else:
            print(f'Q{i}: {row["value"]} - Table too long')

            results.loc[len(results)] = {'question': row["question"], 'value': row["value"], 'response': 'Table too long'}

    os.makedirs('./results', exist_ok=True)
    results.to_csv(f'./results/tapex.csv', index=False)


