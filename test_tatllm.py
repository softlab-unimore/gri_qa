import random
import os
import re
from argparse import ArgumentParser

import pandas as pd
from codecarbon import EmissionsTracker
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed


def create_tables(row, type='one-table'):
    table_dirnames = eval(row["pdf name"])

    tables = []
    for i, table_dirname in enumerate(table_dirnames):
        table_dirname = table_dirname.split(".")[0]
        table_filename = f'dataset/annotation/{table_dirname}/{eval(row["page nbr"])[i]}_{eval(row["table nbr"])[i]}.csv'
        table = pd.read_csv(table_filename, sep=';')
        if type == 'one-table' and 'multitable' not in args.dataset:
            tables.append(table.to_markdown(index=False))
        else:
            company = table_dirname.strip('_2023')
            table = f"Company name: {company}\n\n{table.to_markdown(index=False)}"
            tables.append(table)

    return "\n\n".join(tables)

def create_prompt_step_wise(row, args):
    description = (f"Below is an instruction that describes a question answering task in the environmental domain, paired with "
                   f"{'an input table' if args.type == 'one-table' and 'multitable' not in args.dataset else 'some input tables'} and its relevant text that provide further context. The given question is relevant to "
                   f"the table and text. Generate an appropriate answer to the given question.")

    instruction = ("Given a table and a list of texts in the following, answer the question posed using the following five-step process:\n"
                   "1. Step 1: Predict the type of question being asked. Store this prediction in the variable ‘{question_type}‘. The value of "
                   "‘{question_type}‘ can be one of the following:‘Single span‘, ‘Multiple spans‘, ‘Count‘, or ‘Arithmetic‘.\n"
                   "2. Step 2: Extract the relevant strings or numerical values from the provided table or texts. Store these pieces of evidence "
                   "in the variable ‘{evidence}‘. If there are multiple pieces of evidence, separate them using the ’#’ symbol.\n"
                   "3. Step 3: if the ‘{question_type}‘ is ‘Arithmetic‘, formulate an equation using values stored in ‘{evidence}‘. Store this "
                   "equation in the variable ‘{equation}‘. For all other question types, set the value of {equation} to ’N.A.’.\n"
                   "4. Step 4: Predict or calculate the answer based on the question type, evidence and equation. Store it in the variable "
                   "‘{answer}‘. If there are multiple values, separate them using the ’#’ symbol.\n"
                   "5. Step 5: If the value of the ‘{answer}‘ is numerical, predict its scale and store it in a variable named ‘{scale}‘. The "
                   "value of ‘{scale}‘ can be one of the following: ‘none‘, ‘percent‘, ‘thousand‘, ‘million‘, or ‘billion‘. For non-numerical "
                   "values, set the value of ‘{scale}‘ to ’none’.\n"
                   "Please organize the results in the following table:\n"
                   "| step | output |\n"
                   "| 1 | {question_type} |\n"
                   "| 2 | {evidence} |\n"
                   "| 3 | {equation} |\n"
                   "| 4 | {answer} |\n"
                   "| 5 | {scale} |\n"
                   "Finally, present the final answer in the format: \"The answer is: {answer} #### and its corresponding scale is: {scale}\"")

    tables = create_tables(row, type=args.type)

    return f"{description}\n\n### Instruction\n{instruction}\n\n### Table\n{tables}\n\n### Text\n\n### Question\n{row['question']}\n\n### Response\n"


def create_prompt_end_to_end(row, args):
    description = (f"Below is an instruction that describes a question answering task in the environmental domain, paired with "
                   f"{'an input table' if args.type == 'one-table' and 'multitable' not in args.dataset else 'some input tables'} and its relevant text that provide further context. The given question is relevant to "
                   f"the table{'s' if args.type == 'multi-table' or 'multitable' in args.dataset  else ''} and text{'s' if args == 'multi-table' or 'multitable' in args.dataset else ''}. Generate an appropriate answer to the given question.")

    instruction = ("Given a table and a list of texts in the following, what is the answer to the question? Please predict the answer and store "
                   "it in a variable named ‘{answer}‘. If there are multiple values, separate them using the ’#’ symbol. If the value of the "
                   "‘{answer}‘ is numerical, predict its scale and store it in a variable named ‘{scale}‘. The value of ‘{scale}‘ can be one of "
                   "the following: ‘none‘, ‘percent‘, ‘thousand‘, ‘million‘, or ‘billion‘. For non-numerical values, set the value of ‘{scale}‘ "
                   "to ’none’. Finally, present the final answer in the format of \"The answer is: {answer} #### and its corresponding scale is: {scale}\"")

    tables = create_tables(row, type=args.type)

    return f"{description}\n\n### Instruction\n{instruction}\n\n### Table\n{tables}\n\n### Text\n\n### Question\n{row['question']}\n\n### Response\n"


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='gri-qa_extra.csv')
    parser.add_argument('--end_to_end', action='store_true', default=False)
    parser.add_argument('--type', type=str, default='one-table', choices=['one-table', 'multi-table'], help='Choose if you want raw or norm data')
    args = parser.parse_args()

    set_seed(42)
    random.seed(42)

    qa = pd.read_csv(f'dataset/{args.type}/{args.dataset}', sep=',')
    dataset_name = re.split("[_.]", args.dataset)[1]

    tokenizer = AutoTokenizer.from_pretrained('next-tat/tat-llm-7b-fft')
    model = AutoModelForCausalLM.from_pretrained(
        'next-tat/tat-llm-7b-fft',
        device_map='auto',
    )

    results = pd.DataFrame(columns=['index', 'question', 'value', 'response'])

    os.makedirs(f'./results/{args.type}/{dataset_name}', exist_ok=True)
    tracker = EmissionsTracker(output_dir=f'./results/{args.type}/{dataset_name}')
    tracker.start()

    for i, row in qa.iterrows():

        print(f'Q{i} - {row["question"]}')

        prompt = create_prompt_step_wise(row, args) \
            if not args.end_to_end else create_prompt_end_to_end(row, args)

        encoding = tokenizer(prompt, return_tensors="pt").to(model.device)
        if encoding['input_ids'].shape[1] < 4096:
            outputs = model.generate(**encoding, max_length=4096)
            response = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            try:
                response_value = response[0].split('The answer is: ')[2].split(' ###')[0]
                print(f'Q{i}: {row["value"]} - {response_value}')
            except:
                response_value = "No answer in the response"
                print(f'Q{i}: {row["value"]} - No answer in the response')
        else:
            response_value = "Response too long"

        results.loc[len(results)] = {'index': i, 'question': row["question"], 'value': row["value"], 'response': response_value}

    tracker.stop()

    results.to_csv(f'./results/{args.type}/{dataset_name}/tatllm__{"step_wise" if not args.end_to_end else "end_to_end"}.csv', index=False)
    os.rename(f'./results/{args.type}/{dataset_name}/emissions.csv',f'./results/{args.type}/{dataset_name}/emissions_{"step_wise" if not args.end_to_end else "end_to_end"}.csv')
