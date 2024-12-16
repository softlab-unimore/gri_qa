import os
from argparse import ArgumentParser

import pandas as pd


def check_number(value, response, percentage=False):
    try:
        check = False
        if percentage:
            response_norm = float(response) * 100
            check = float(value) == float(response_norm)
        return float(value) == float(response) or check
    except ValueError:
        return False


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='extra', choices=['extra', 'quant', 'rel', 'kw', 'neg'])
    args = parser.parse_args()

    # models = ['tatllm', 'tapex', 'tablellama', 'finma', 'tagop', 'openai']
    models = ['tatllm', 'tapex', 'tablellama', 'finma']

    metrics = pd.DataFrame(columns=['model', 'em'])

    os.makedirs(f'./results/{args.dataset}/with_match', exist_ok=True)

    for model in models:
        print(f'--> Processing {model}')

        results = pd.read_csv(f'./results/{args.dataset}/{model}.csv')
        results['value'] = results['value'].astype(str).str.lower()
        results['response'] = results['response'].astype(str).str.lower()

        for i, row in results.iterrows():
            row['response'] = row['response'].strip(' |')

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
            elif any(r.isalpha() for r in row['response']):
                el = [c.strip('%') for c in row['response'].split(' ')]
                for e in el:
                    if check_number(row['value'].strip('%'), e, percentage=percentage):
                        results.loc[i, 'correct'] = True
                        break
                    results.loc[i, 'correct'] = False

            # Check for numbers with ., % symbols and without <, =, > symbols
            elif (any(c in row['value'] or c in row['response'] for c in ['.', '%']) and
                  (any(c in row['value'] for c in ['<', '=', '>']) == any(c in row['response'] for c in ['<', '=', '>']))):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%<= >'), row['response'].strip('%<= >'), percentage=percentage)

            # Otherwise
            else:
                results.loc[i, 'correct'] = False

        results.to_csv(f'./results/{args.dataset}/with_match/{model}.csv', index=False)
        em = results.loc[results['correct'] == True].shape[0] / results.shape[0]
        metrics.loc[len(metrics)] = {'model': model, 'em': round(em, 5)}
        print(f'EM: {round(em, 5)}')

    metrics.to_csv(f'./results/{args.dataset}/metrics.csv', index=False)
