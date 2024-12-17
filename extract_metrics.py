import os
from argparse import ArgumentParser

import pandas as pd


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
    parser.add_argument('--dataset', type=str, default='extra', choices=['extra', 'quant', 'rel', 'kw', 'neg'])
    args = parser.parse_args()

    # models = ['tatllm', 'tapex', 'tablellama', 'finma', 'tagop', 'openai']
    # models = ['tatllm__end_to_end', 'tapex', 'tablellama', 'finma', 'openai']
    models = ['tablellama']

    metrics = pd.DataFrame(columns=['model', 'em'])

    os.makedirs(f'./results/{args.dataset}/with_match', exist_ok=True)

    for model in models:
        print(f'--> Processing {model}')

        results = pd.read_csv(f'./results/{args.dataset}/{model}.csv')
        results['value'] = results['value'].astype(str).str.lower()
        results['response'] = results['response'].astype(str).str.lower()

        for i, row in results.iterrows():

            row['response'] = row['response'].strip(' |()')

            percentage = True if '%' in row['question'] or 'percentage' in row['question'] else False

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
            elif (any(r.isalpha() for r in row['response'])) or (any(c in row['response'] for c in ['(', ')'])):
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
