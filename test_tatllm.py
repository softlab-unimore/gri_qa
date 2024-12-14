import os
import re
from argparse import ArgumentParser

import pandas as pd
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM


def create_prompt(table, question):
    description = ("Below is an instruction that describes a question answering task in the environmental domain, paired with "
                   "an input table and its relevant text that provide further context. The given question is relevant to "
                   "the table and text. Generate an appropriate nswer to the given question.")

    instruction = ("Given a table and a list of texts in the following, what is the answer to the question? Please predict the answer and store "
                   "it in a variable named ‘{answer}‘. If there are multiple values, separate them using the ’#’ symbol. If the value of the "
                   "‘{answer}‘ is numerical, predict its scale and store it in a variable named ‘{scale}‘. The value of ‘{scale}‘ can be one of "
                   "the following: ‘none‘, ‘percent‘, ‘thousand‘, ‘million‘, or ‘billion‘. For non-numerical values, set the value of ‘{scale}‘ "
                   "to ’none’. Finally, present the final answer in the format of \"The answer is: {answer} #### and its corresponding scale is: {scale}\"")

    table = table.to_markdown(index=False)

    return f"{description}\n\n###Instruction:\n{instruction}\n\n###Table:\n{table}\n\n###Text\n\n###Question:\n{question}\n\n###Response:\n"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gri-qa_extra.csv')
    args = parser.parse_args()

    qa = pd.read_csv(f'dataset/{args.dataset}', sep=',')
    qa = qa[qa.iloc[:, 2] != 2.0]
    dataset_name = re.split("[_.]", args.dataset)[1]

    tokenizer = AutoTokenizer.from_pretrained('next-tat/tat-llm-7b-fft')
    model = AutoModelForCausalLM.from_pretrained(
        'next-tat/tat-llm-7b-fft',
        device_map='auto',
    )

    results = pd.DataFrame(columns=['index', 'question', 'value', 'response'])

    tracker = EmissionsTracker(output_dir=f'./results/{dataset_name}')
    tracker.start()

    for i, row in qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        # Table extraction
        table_dirname = row["pdf name"].split('.')[0]
        table_filename = f'dataset/annotation/{table_dirname}/{row["page nbr"]}_{row["table nbr"]}.csv'
        table = pd.read_csv(table_filename, sep=';')

        prompt = create_prompt(table, row["question"])

        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(**encoding, max_length=4096)
        response = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        try:
            response_value = response[0].split('The answer is: ')[
                2].split(' ###')[0]
            print(f'Q{i}: {row["value"]} - {response_value}')
        except:
            response_value = "No answer in the response"
            print(f'Q{i}: {row["value"]} - No answer in the response')

        results.loc[len(results)] = {
            'index': i, 'question': row["question"], 'value': row["value"], 'response': response_value}

    tracker.stop()

    os.makedirs(f'./results/{dataset_name}', exist_ok=True)
    results.to_csv(f'./results/{dataset_name}/tatllm.csv', index=False)
    os.rename(f'./results/{dataset_name}/emissions.csv',
              f'./results/{dataset_name}/emissions_tatllm.csv')
