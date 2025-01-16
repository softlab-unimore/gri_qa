import pandas as pd
import os
from ast import literal_eval

def mapper(row):
    pdfs, page_nbrs, table_nbrs = [], [], []
    for pdf in literal_eval(row["pdf name"]):
        splitted = pdf.split("_")
        pdfs.append('_'.join(splitted[:-2]))
        page_nbrs.append(splitted[-2])
        table_nbrs.append(splitted[-1])

    if pd.isna(row["answer_company"]) or row["answer_company"].strip() == "[]":
        answer_company_list = []
    elif row["answer_company"].strip()[0] == "[":
        answer_company_list = row["answer_company"].strip().strip("[]").replace("'", "").split()
    else:
        answer_company_list = [row["answer_company"]]

    row["pdf name"] = pdfs
    row["page nbr"] = page_nbrs
    row["table nbr"] = table_nbrs
    row["answer_company"] = answer_company_list

    return row


for file_name in os.listdir("./"):
    if file_name.split(".")[-1] != "csv":
        continue
    df = pd.read_csv(file_name)
    df = df.rename(columns={"companies": "pdf name", "operation": "question_type_ext", "GRI": "gri", "out": "value"})
    df.insert(2, "table nbr", [""]*len(df))
    df.insert(2, "page nbr", [""]*len(df))
    df = df.apply(mapper, axis=1)
    df.to_csv(file_name, index=False)
