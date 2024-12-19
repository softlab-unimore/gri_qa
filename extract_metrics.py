import os
import re
from argparse import ArgumentParser

import pandas as pd

def preprocessing_range_tablellama(row):
    row['response'] = re.sub(r'[<>]', '', row['response'])
    row['response'] = re.sub(r'[,]', ' -', row['response'])
    row['response'] = re.sub(r'(\d+(\.\d+)?)-(\d+(\.\d+)?)', r'\1 - \3', row['response'])
    row['response'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.1f}", row['response'])
    row['value'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.1f}", row['value'])
    return row

def preprocessing_range_finma(row):
    row['response'] = row['response'].replace('and', '-')
    row['response'] = re.sub(r'(\d+(\.\d+)?)[-–—](\d+(\.\d+)?)', r'\1 - \3', row['response'])
    return row

def preprocessing_range_tattllm__end_to_end(row, dataset):
    if dataset == 'extra':
        row['response'] = row['response'].replace('#', ' - ')
        row['value'] = re.sub(r'[-–—]', '-', row['value'])
    elif dataset == 'rel':
        row['response'] = row['response'].replace('#', ', ')
        row['response'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.1f}", row['response'])
        row['value'] = re.sub(r'\d+(\.\d+)?', lambda m: f"{float(m.group()):.1f}", row['value'])
    return row

def check_number(value, response, percentage=False):
    try:
        if percentage:
            if (abs(float(value)) == abs(float(response)) or
                    abs(float(value)) == abs(float(float(response) * 100))):
                return True
        return float(value) == float(response)
    except ValueError:
        return False


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='rel', choices=['extra', 'quant', 'rel', 'kw', 'neg'])
    args = parser.parse_args()

    # models = ['tatllm', 'tapex', 'tablellama', 'finma', 'tagop', 'openai']
    models = ['tatllm__end_to_end', 'tapex', 'tablellama', 'finma', 'openai', 'openai_chainofthought']
    # models = ['openai_chainofthought']

    metrics = pd.DataFrame(columns=['model', 'em'])

    os.makedirs(f'./results/{args.dataset}/with_match', exist_ok=True)

    for model in models:
        print(f'--> Processing {model}')

        results = pd.read_csv(f'./results/{args.dataset}/{model}.csv')
        results['value'] = results['value'].astype(str).str.lower()
        results['response'] = results['response'].astype(str).str.lower()

        for i, row in results.iterrows():

            # if row['index'] == 192:
            row['response'] = row['response'].split('###')[0]
            row['response'] = row['response'].strip(' |()\n\r')

            percentage = True if '%' in row['question'] or 'percentage' in row['question'] else False

            if args.dataset == 'extra':
                range = True if re.match(r'^\d+(\.\d+)?\s*[-–—]\s*\d+(\.\d+)?$', row['value']) else False
            else: # rel dataset
                range = True if re.match(r'^\s*\d+(\.\d+)?\s*,\s*\d+(\.\d+)?(\s*,\s*\d+(\.\d+)?)*\s*$', row['value']) else False

            # Check for range
            if range:
                row['value'] = re.sub(r'(\d+(\.\d+)?)[-–—](\d+(\.\d+)?)', r'\1 - \3', row['value'])
                if model == 'tablellama':
                    row = preprocessing_range_tablellama(row)
                if model == 'finma':
                    row = preprocessing_range_finma(row)
                if model == 'tatllm__end_to_end':
                    row = preprocessing_range_tattllm__end_to_end(row, args.dataset)

            # Check extact match
            if row['value'] == row['response']:
                results.loc[i, 'correct'] = True

            # Check for numbers with <, =, > symbols in both side
            elif row['value'].startswith('<=') and row['response'].startswith('<='):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%<= '), row['response'].strip('%<= '), percentage=percentage)
            elif row['value'].startswith('>=') and row['response'].startswith('>='):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%>= '), row['response'].strip('%>= '), percentage=percentage)
            elif row['value'].startswith('<') and row['response'].startswith('<'):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%< '), row['response'].strip('%< '), percentage=percentage)
            elif row['value'].startswith('=') and row['response'].startswith('='):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%= '), row['response'].strip('%= '), percentage=percentage)
            elif row['value'].startswith('>') and row['response'].startswith('>'):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%> '), row['response'].strip('%> '), percentage=percentage)

            # Check for response with multiple words
            elif ((any(r.isalpha() for r in row['response'])) or (any(c in row['response'] for c in ['(', ')']))) and args.dataset == 'extra':
                el = [c.strip('%()') for c in row['response'].split(' ')]
                results.loc[i, 'correct'] = any(
                    check_number(row['value'].strip('%'), e, percentage=percentage) for e in el
                )
                if any(row['value'] in e for e in el):
                    results.loc[i, 'correct'] = True


            # Check for numbers with ., % symbols and without <, =, > symbols
            elif (((any(c in row['value'] or c in row['response'] for c in ['.', '%']) and
                  (any(c in row['value'] for c in ['<', '=', '>']) == any(c in row['response'] for c in ['<', '=', '>'])))) or
                  any(c in row['value'] or c in row['response'] for c in ['~', ','])):
                row['response'] = row['response'].replace(',', '')
                results.loc[i, 'correct'] = check_number(row['value'].strip('%<= >~'), row['response'].strip('%<= >~'), percentage=percentage)

            # Otherwise
            else:
                results.loc[i, 'correct'] = False

        results.to_csv(f'./results/{args.dataset}/with_match/{model}.csv', index=False)
        em = results.loc[results['correct'] == True].shape[0] / results.shape[0]
        metrics.loc[len(metrics)] = {'model': model, 'em': round(em, 5)}
        print(f'EM: {round(em, 5)}')

    metrics.to_csv(f'./results/{args.dataset}/metrics.csv', index=False)
