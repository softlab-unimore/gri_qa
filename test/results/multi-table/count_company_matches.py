import pandas as pd
import os

def is_float(num):
    try:
        float(num)
        return True
    except:
        return False

template = "How many {} correct questions?"
template_sup = "-> of which {} superlative"
template_rank = "-> of which {} ranking"

for dir_name in os.listdir('./'):
    if "quant" in dir_name:
        continue
    if not os.path.isdir(dir_name):
        continue

    path = f"./{dir_name}/with_match/openai_chainofthought.csv" #_chainofthought.csv"
    df = pd.read_csv(path)
    total_length = len(df)
    count_sup, count_rank = 0, 0
    total_correct = len(df[df["correct"] == True])
    total_company = 0
    for i, row in df[df["correct"] == True].iterrows():
        if is_float(row["response"]):
            continue
        response_splitted = row["response"].split(",")
        if len(response_splitted) > 1:
            if is_float(response_splitted[0]):
                continue
            count_rank += 1
        else:
            count_sup += 1

    for i,row in df.iterrows():
        if is_float(row["response"]):
            continue
        response_splitted = row["response"].split(",")
        if len(response_splitted) > 1:
            if is_float(response_splitted[0]):
                continue
        total_company += 1

    print(template.format(dir_name), total_correct)
    print(template_sup.format(count_sup))
    print(template_rank.format(count_rank))
    print("Total company answers:", total_company)
    print("Total length of dataset:", total_length)