import pandas as pd

df = pd.read_csv("gri-qa_extra.csv")
df = df.drop(["checked", "Unnamed: 0"], axis="columns")
df = df[df["hierarchical"] != 2]
df["hierarchical"] = df["hierarchical"].fillna(0)
df["hierarchical"] = df["hierarchical"].map({0:0, 1:0, 3:1})
print(df.head())
print(df["hierarchical"].unique())
df = df.reset_index(drop=True)
df.to_csv("gri-qa_extra2.csv", index=True)