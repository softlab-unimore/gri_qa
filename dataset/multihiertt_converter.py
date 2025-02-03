import pandas as pd
from table_description_generation import *
import json
from tqdm import tqdm

class Wrapper:
    def __init__(self, df):
        self.df = df
    def __eq__(self, other):
        return self.df.equals(other.df)
    def __hash__(self):
        return hash((tuple(self.df.index), tuple([tuple(el) for el in self.df.values.tolist()])))

df = pd.read_csv("one-table/gri-qa_extra.csv")
results_json = []

for i,row in tqdm(df.iterrows()):
    tables = []
    table_descriptions = {}
    for j, (pdf_name, page_nbr, table_nbr) in enumerate(zip(eval(row["pdf name"]), eval(row["page nbr"]), eval(row["table nbr"]))):
        pdf_name = pdf_name.split(".")[0]
        path = f"./annotation/{pdf_name}/{page_nbr}_{table_nbr}.csv"
        table = pd.read_csv(path, sep=";")
        tables.append(table.to_html())
        table_wrapped = Wrapper(table)
        table_descriptions |= generateDescription(table_wrapped, tuple(table.columns), top_header_nonexist_flag=0, num_table=j)

    new_sample = {
        "uid": str(i),
        "paragraphs": [""],
        "tables": tables,
        "table_description": table_descriptions,
        "qa": {
            "question": row["question"],
            "answer": row["value"],
            "program": "",
            "text_evidence": "",
            "table_evidence": ""
        }
    }

    results_json.append(new_sample)

with open("multihiertt_extra.json", "w") as file:
    json.dump(results_json, file, indent=4)