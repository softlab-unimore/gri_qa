import pandas as pd


def remove_trailing_zeros(num_str):
    if '.' in num_str:
        num_str = num_str.rstrip('0').rstrip('.')
    return num_str

def remove_perc(num_str):
    num_str = num_str.rstrip('%')
    return num_str

dataset = pd.read_csv("../dataset/qa_dataset.csv", sep=",")
dataset["Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)"] = dataset["Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)"].fillna(0.0)

dataset = dataset[dataset["Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)"].isin([0.0,1.0,3.0])]
dataset = dataset[dataset["row"].notna()]

total_errors = 0
for i,row in dataset.iterrows():
    path = f"../dataset/annotation/{'.'.join(row['pdf name'].split('.')[:-1])}/{row['page nbr']}_{row['table nbr']}.csv"
    table = pd.read_csv(path, sep=";")
    gold = remove_trailing_zeros(remove_perc(str(row['value']).strip()))
    try:
        from_idx = remove_trailing_zeros(remove_perc(str(table.iloc[int(row['row'])-2, int(row['column'])-1]).strip()))
    except:
        total_errors += 1
        print(f"{path} --- {row['value']}")
        continue
    
    if gold != from_idx:
        total_errors += 1
        print(f"{path} --- {row['value']} --- {table.iloc[int(row['row'])-2, int(row['column'])-1]}")

print(total_errors)
    
    