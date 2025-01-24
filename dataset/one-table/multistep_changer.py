import re

import pandas as pd
import random
import string

def replace_last_occurrence(s, old, new):
    parts = s.rsplit(old, 1)
    return new.join(parts)

def question_changer(row):
    if row["question_type_ext"] in ["ratio_average", "average_sum", "sum_average"]:
        row["question"] = row["question"].strip()
        if row["question"][0].lower() == "b":
            if "but" in row["question"]:
                #between = ', '.join(row["question"].lower().split("what")[0].split("which")[0].split(", ")[:2])
                #rest = ', '.join(row["question"].split(", ")[2:])
                if "what" in row["question"]:
                    between = row["question"].lower().split("what")[0]
                    rest = row["question"].lower().split("what")[1]
                    rest = "what" + rest
                else:
                    between = row["question"].lower().split("which")[0]
                    rest = row["question"].lower().split("which")[1]
                    rest = "which" + rest
            else:
                if "what" in row["question"]:
                    between = row["question"].lower().split("what")[0]
                    rest = row["question"].lower().split("what")[1]
                    rest = "what" + rest
                else:
                    between = row["question"].lower().split("which")[0]
                    rest = row["question"].lower().split("which")[1]
                    rest = "which" + rest
        else:
            rest, between = 'between '.join(row["question"].split("between ")[:-1]), row["question"].split("between ")[-1]
            between = "between " + between[:-1]
            rest = rest.strip() + "?"

        between = between.strip().rstrip(string.punctuation)
        incipit = ' '.join(rest.split()[:4])
        row["question"] = f"{incipit[0].upper()}{incipit[1:]} {between[0].lower()}{between[1:]} {' '.join(rest.split()[4:])}"

    elif row["question_type_ext"] in ["sum_ranking", "average_ranking", "ratio_ranking"]:
        row["question"] = row["question"].strip()
        if row["question"][0].lower() == "b":
            if "but" in row["question"]:
                if "what" in row["question"]:
                    between = row["question"].lower().split("what")[0]
                    rest = row["question"].lower().split("what")[1]
                    rest = "what" + rest
                else:
                    between = row["question"].lower().split("which")[0]
                    rest = row["question"].lower().split("which")[1]
                    rest = "which" + rest
            else:
                if "what" in row["question"]:
                    between = row["question"].lower().split("what")[0]
                    rest = row["question"].lower().split("what")[1]
                    rest = "what" + rest
                else:
                    between = row["question"].lower().split("which")[0]
                    rest = row["question"].lower().split("which")[1]
                    rest = "which" + rest
        else:
            rest, between = 'between '.join(row["question"].split("between ")[:-1]), row["question"].split("between ")[-1]
            between = "between " + between[:-1]
            rest = rest.strip() + "?"

        between = between.strip().rstrip(string.punctuation)
        incipit = ' '.join(rest.split()[:6])
        row["question"] = f"{incipit[0].upper()}{incipit[1:]} values {between[0].lower()}{between[1:]} of the {' '.join(rest.split()[6:])}"

    else:
        row["question"] = row["question"].strip()
        if row["question"][0].lower() == "b":
            if "but" in row["question"]:
                if "what" in row["question"]:
                    between = row["question"].lower().split("what")[0]
                    rest = row["question"].lower().split("what")[1]
                    rest = "what" + rest
                else:
                    between = row["question"].lower().split("which")[0]
                    rest = row["question"].lower().split("which")[1]
                    rest = "which" + rest
            else:
                if "what" in row["question"]:
                    between = row["question"].lower().split("what")[0]
                    rest = row["question"].lower().split("what")[1]
                    rest = "what" + rest
                else:
                    between = row["question"].lower().split("which")[0]
                    rest = row["question"].lower().split("which")[1]
                    rest = "which" + rest
        else:
            rest, between = 'between '.join(row["question"].split("between ")[:-1]), row["question"].split("between ")[-1]
            between = "between " + between[:-1]
            rest = rest.strip() + "?"

        between = between.strip().rstrip(string.punctuation)
        incipit = ' '.join(rest.split()[:4])
        row["question"] = f"{incipit[0].upper()}{incipit[1:]} {between[0].lower()}{between[1:]} {' '.join(rest.split()[4:])}"

    if " from " in row["question"] and " to " in row["question"]:
        row["question"] = replace_last_occurrence(row["question"], " from ", " between ")
        row["question"] = replace_last_occurrence(row["question"], " to ", " and ")
    if row["question_type_ext"] == "sum_average":
        row["question"] = replace_last_occurrence(row["question"], " to ", " and ")

    if row["question_type_ext"] in ["average_sup", "sum_ranking", "average_ranking", "sum_average", "average_sum"]:
        row["question"] = replace_last_occurrence(row["question"], " in ", " between ")

    row["question"] = re.sub(' +', ' ', row["question"])

    return row

df = pd.read_csv("gri-qa_multistep.csv")
new_dataset = []
for _, row in df.iterrows():
    new_dataset.append(question_changer(row))

pd.DataFrame(new_dataset, columns=["pdf name", "page nbr", "table nbr", "gri", "question", "question_type_ext", "value", "row indices", "col indices", "metadata"]).to_csv("gri-qa_multistep_new.csv", index=False)