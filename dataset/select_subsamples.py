import pandas as pd
import os
import numpy as np

def columns_to_select(df, cols):
    for col in cols:
        if col not in df:
            df[col] = np.nan

    return df[cols]

extra = pd.read_csv("./one-table/gri-qa_extra.csv")
rel = pd.read_csv("./one-table/gri-qa_rel.csv")
quant = pd.read_csv("./one-table/gri-qa_quant.csv")
step = pd.read_csv("./one-table/gri-qa_multistep.csv")
m2_step = pd.read_csv("./multi-table/gri-qa_multitable2_multistep.csv")
m3_step = pd.read_csv("./multi-table/gri-qa_multitable3_multistep.csv")
m5_step = pd.read_csv("./multi-table/gri-qa_multitable5_multistep.csv")
m2_rel = pd.read_csv("./multi-table/gri-qa_multitable2_rel.csv")
m3_rel = pd.read_csv("./multi-table/gri-qa_multitable3_rel.csv")
m5_rel = pd.read_csv("./multi-table/gri-qa_multitable5_rel.csv")
m2_quant = pd.read_csv("./multi-table/gri-qa_multitable2_quant.csv")
m3_quant = pd.read_csv("./multi-table/gri-qa_multitable3_quant.csv")
m5_quant = pd.read_csv("./multi-table/gri-qa_multitable5_quant.csv")

hier, extra = extra[extra["hierarchical"] == 1], extra[extra["hierarchical"] == 0]

os.makedirs("./samples", exist_ok=True)

extra = columns_to_select(extra, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
hier = columns_to_select(hier, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
rel = columns_to_select(rel, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
quant = columns_to_select(quant, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
step = columns_to_select(step, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m2_step = columns_to_select(m2_step, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m3_step = columns_to_select(m3_step, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m5_step = columns_to_select(m5_step, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m2_rel = columns_to_select(m2_rel, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m3_rel = columns_to_select(m3_rel, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m5_rel = columns_to_select(m5_rel, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m2_quant = columns_to_select(m2_quant, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m3_quant = columns_to_select(m3_quant, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])
m5_quant = columns_to_select(m5_quant, ["pdf name", "page nbr", "table nbr", "gri", "gri_finegrained", "question_type_ext", "question", "value"])


num_samples = {
    "extra": (extra, 50),
    "hier": (hier, 50),
    "quant": (quant, 50),
    "rel": (rel, 50),
    "step": (step, 50),
    "multitable2_multistep": (m2_step, 17),
    "multitable3_multistep": (m3_step, 17),
    "multitable5_multistep": (m5_step, 16),
    "multitable2_rel": (m2_rel, 17),
    "multitable3_rel": (m3_rel, 17),
    "multitable5_rel": (m5_rel, 16),
    "multitable2_quant": (m2_quant, 17),
    "multitable3_quant": (m3_quant, 17),
    "multitable5_quant": (m5_quant, 16),
}

final_df = pd.DataFrame()
#final_df["original_dataset"] = None

for k, (v, num) in num_samples.items():
    samples_df = v.sample(num, random_state=42)
    samples_df["original_dataset"] = [k]*len(samples_df)

    for i, row in samples_df.iterrows():
        pdf_names, page_nbrs, table_nbrs = eval(row["pdf name"]), eval(row["page nbr"]), eval(row["table nbr"])
        csv_contents = ""
        for pdf_name, page_nbr, table_nbr in zip(pdf_names, page_nbrs, table_nbrs):
            company_name = '_'.join(pdf_name.split("_")[:-1])
            dir_name = pdf_name.split(".")[0]
            table = pd.read_csv(f"./annotation/{dir_name}/{page_nbr}_{table_nbr}.csv", sep=";")
            csv_contents += f"<h3>TABLE of COMPANY {company_name}</h3><br>{table.to_html()}<br><br>"
        samples_df.at[i,"csv_contents"] = csv_contents
    final_df = pd.concat([final_df, samples_df])
    samples_df = samples_df.reset_index()
    samples_df = samples_df.drop("csv_contents", axis="columns")
    samples_df.to_csv(f"./samples/samples_gri-qa_{k}.csv", index=False)

final_df = final_df.reset_index()

target = final_df["value"]
final_df = final_df.drop("value", axis="columns")
final_df.to_csv("./samples/samples_gri-qa.csv", index=True)
target.to_csv("./samples/targets_gri-qa.csv", index=True)