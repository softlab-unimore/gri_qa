from argparse import ArgumentParser

import pandas as pd


def check_number(value, response):
    try:
        return float(value) == float(response)
    except ValueError:
        return False


if __name__=='__main__':

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='extra', choices=['extra', 'quant', 'rel', 'keyw', 'neg'])
    args = parser.parse_args()

    # models = ['tatllm', 'tapex', 'tablellama', 'finma', 'tagop', 'gpt-4']
    models = ['tatllm', 'tapex', 'tablellama', 'finma']

    for model in models:
        print(f'Processing {model}')

        results = pd.read_csv(f'./results/{args.dataset}/{model}.csv')
        results['value'] = results['value'].astype(str).str.lower()
        results['response'] = results['response'].astype(str).str.lower()

        for i, row in results.iterrows():
            row['response'] = row['response'].strip(' |')

            # Check extact match
            if row['value'] == row['response']:
                results.loc[i, 'correct'] = True

            # Check for numbers with <, =, > symbols in both side
            elif row['value'].startswith('<') and row['response'].startswith('<'):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%< '), row['response'].strip('%< '))
            elif row['value'].startswith('=') and row['response'].startswith('='):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%= '), row['response'].strip('%= '))
            elif row['value'].startswith('>') and row['response'].startswith('>'):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%> '), row['response'].strip('%> '))
            elif row['value'].startswith('<=') and row['response'].startswith('<='):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%<= '), row['response'].strip('%<= '))
            elif row['value'].startswith('>=') and row['response'].startswith('>='):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%>= '), row['response'].strip('%>= '))

            # Check for response with multiple words
            elif any(r.isalpha() for r in row['response']):
                el = [c.strip('%') for c in row['response'].split(' ')]
                for e in el:
                    if check_number(row['value'].strip('%'), e):
                        results.loc[i, 'correct'] = True
                        break
                    results.loc[i, 'correct'] = False

            # Check for numbers with ., % symbols and without <, =, > symbols
            elif (any(c in row['value'] or c in row['response'] for c in ['.', '%']) and
                  any(c not in row['value'] and c not in row['response'] for c in ['<', '=', '>'])):
                results.loc[i, 'correct'] = check_number(row['value'].strip('%<= >'), row['response'].strip('%<= >'))

            # Otherwise
            else:
                results.loc[i, 'correct'] = False

            print(f'{row["value"]} - {row["response"]}')
            print(f'Correct: {results.loc[i, "correct"]}')
