import pandas as pd
import json

df = pd.read_csv("gri-qa_extra.csv")
used_gris = df["gri_finegrained"].unique()
print(used_gris)
filtered_gris = []
for i in range(len(used_gris)):
	if str(used_gris[i]) == "?" or str(used_gris[i]) == "nan":
		continue
	filtered_gris.append(used_gris[i])

with open("en_queries_extended.json") as f:
	gri_json = json.load(f)

filtered_gris_json = {}
for k,v in gri_json.items():
	if k not in filtered_gris:
		continue
	filtered_gris_json[k] = v

with open("en_gris.json", "w") as f:
	json.dump(filtered_gris_json,f)
