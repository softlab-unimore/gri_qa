import pandas as pd

df = pd.read_csv("gri-qa_extra3.csv")
df = df[df["Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)"] != 2]
df = df.dropna(subset=["row", "column"])

result = df.groupby('gri_finegrained').apply(
    lambda group: list(group[["gri", "gri_finegrained", "pdf name", "page nbr", "table nbr", "row", "column"]].drop_duplicates(subset=["pdf name", "page nbr", "table nbr"]).apply(tuple, axis=1).to_dict().values())
)

result = result.to_dict()
dict2, dict3, dict5, dict10 = {}, {}, {}, {}

for k,v in result.items():
    pdf_names = set([value[2] for value in v])
    if len(pdf_names) >= 2:
        dict2[k] = v
    if len(pdf_names) >= 3:
        dict3[k] = v
    if len(pdf_names) >= 5:
        dict5[k] = v
    if len(pdf_names) >= 10:
        dict10[k] = v

print(dict2)
print(len(dict2.keys()))