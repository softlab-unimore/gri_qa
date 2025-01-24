import pandas as pd
import random

from tqdm import tqdm


class DefaultDict(dict):
    def __missing__(self, key):
        return f"{{{key}}}"

operations = [
    "average_sup", "sum_sup", "ratio_sup",
    "sum_ranking", "average_ranking", "ratio_ranking",
    "ratio_average", "sum_average",
    "average_sum"
]

templates = {
    "average_sup": "Between {topics}, what is the value of the {hl} average in the years {years}?",
    "sum_sup": "Between {topics}, what is the value of the {hl} sum between the years {years}?",
    "ratio_sup": "Between {topics}, what is the value of the {hl} percentage ratio from {years}?",
    "sum_ranking": "Between {topics}, what are the top {k} {hl} sums (in {order} order) in the years {years}?",
    "average_ranking": "Between {topics}, what are the top {k} {hl} averages (in {order} order) in the years {years}?",
    "ratio_ranking": "Between {topics}, what are the top {k} {hl} percentage ratios (in {order} order) from the year {years}?",
    "ratio_average": "Between {topics}, what is the average of the percentage ratios from {years}?",
    "sum_average": "Between {topics}, what is the average of the sums in the years {years}?",
    "average_sum": "Between {topics}, what is the sum of the averages in the years {years}?"
}

def count_numbers(lst):
    count = 0
    columns = []
    for i, el in enumerate(lst):
        try:
            float(el)
            count+=1
            columns.append(el)
        except:
            pass
    return count, columns

def choose_operation(row, columns):
    num_numbers, num_columns = count_numbers(columns)
    if num_numbers < 2:
        return None, None

    metadata = {}
    if isinstance(row["col indices"], str):
        row["col indices"] = eval(row["col indices"])

    op = random.sample(operations, 1)[0]
    template = templates[op]
    metadata["operation"] = op
    col_position = []

    if op in ["average_sup", "sum_sup", "sum_ranking", "average_ranking", "average_sum"]:
        num_years = random.randint(2, num_numbers)

        years_sampled = random.sample(num_columns, num_years)
        for year in years_sampled:
            col_position.append(list(columns).index(year)+1)
        years = ", ".join(years_sampled)
    else:
        num_years = 2
        years_sampled = random.sample(num_columns, num_years)
        for year in years_sampled:
            col_position.append(list(columns).index(year)+1)
        years = " to ".join(years_sampled)

    metadata["years"] = years_sampled
    metadata["col_position"] = col_position

    if op in ["average_sup", "sum_sup", "ratio_sup", "sum_ranking", "average_ranking", "ratio_ranking"]:
        hl = random.sample(["highest", "lowest"], 1)[0]
        metadata["hl"] = hl
    if op in ["sum_ranking", "average_ranking", "ratio_ranking"]:
        order = random.sample(["ascending", "descending"], 1)[0]
        k = random.randint(2, num_years)
        metadata["k"] = k
        metadata["order"] = order

    if op in ["average_sup", "sum_sup", "ratio_sup"]:
        tmp = {"hl": hl, "years": years}
        prompt = template.format_map(DefaultDict(tmp))
    elif op in ["sum_ranking", "average_ranking", "ratio_ranking"]:
        tmp = {"k": k, "hl": hl, "order": order, "years": years}
        prompt = template.format_map(DefaultDict(tmp))
    else:
        tmp = {"years": years}
        prompt = template.format_map(DefaultDict(tmp))

    return prompt, metadata

def get_sum(table, r_idx, col_names):
    try:
        for name in col_names:
            if pd.isna(table.at[r_idx-2,name]):
                return None
        return round(sum([float(table.at[r_idx-2,name]) for name in col_names]),2)
    except:
        return None

def get_average(table, row_idx, col_names):
    sum = get_sum(table, row_idx, col_names)
    if sum is None:
        return None
    return round(sum/len(col_names),2)

def get_ratio(table, row_idx, col_names):
    if len(col_names) != 2:
        return None
    try:
        if float(table.at[row_idx-2,col_names[1]]) == 0 or pd.isna(table.at[row_idx-2,col_names[1]]):
            return None
        return round(float(table.at[row_idx-2,col_names[0]])/float(table.at[row_idx-2,col_names[1]]),4)*100
    except:
        return None

def get_value(row, table, metadata):
    row_idx = eval(row["row indices"])
    col_names = metadata["years"]
    aggregation_op = metadata["operation"].split("_")[0]
    final_operation = metadata["operation"].split("_")[1]
    values = []
    for r_idx in row_idx:
        if aggregation_op == "average":
            values.append(get_average(table, r_idx, col_names))
        elif aggregation_op == "sum":
            values.append(get_sum(table, r_idx, col_names))
        elif aggregation_op == "ratio":
            values.append(get_ratio(table, r_idx, col_names))
        else:
            raise ValueError("Invalid operation")

    if None in values:
        return None, None, None
    
    row_pos, col_pos = [[r_idx]*len(metadata["col_position"]) for r_idx in row_idx], metadata["col_position"]*len(row_idx)
    if final_operation == "sup":
        if metadata["hl"] == "highest":
            fn = max
        else:
            fn = min
        value = round(fn(values),2)
    elif final_operation == "average":
        value = round(sum(values)/len(values),2)
    elif final_operation == "ranking":
        s_values = sorted(values)
        if metadata["hl"] == "lowest":
            s_values = s_values[::-1]
        s_values = s_values[-metadata["k"]:]
        if metadata["order"] == "ascending":
            s_values = sorted(s_values)
        elif metadata["order"] == "descending":
            s_values = sorted(values, reverse=True)
        value = ', '.join([str(round(s,2)) for s in s_values])
    elif final_operation == "sum":
        value = round(sum(values),2)
    else:
        raise ValueError("Invalid operation")
    return value, row_pos, col_pos

if __name__ == "__main__":
    df = pd.read_csv("gri-qa_quant.csv")
    df = df[df["row/column spanning"] == 1]
    df.reset_index(inplace=True)

    new_dataset = []
    op_already_seen = []
    for i in range(3):
        for j,row in tqdm(df.iterrows()):
            if i == 0:
                op_already_seen.append([])
            op = ""
            while op == "" or op in op_already_seen[j]:
                pdf_name, page_nbr, table_nbr = eval(row["pdf name"])[0].split(".")[0], eval(row["page nbr"])[0], eval(row["table nbr"])[0]
                path = f"../annotation/{pdf_name}/{page_nbr}_{table_nbr}.csv"
                table = pd.read_csv(path, sep=";")
                prompt, metadata = choose_operation(row, table.columns)
                op = metadata["operation"]

                if prompt is None:
                    break
            if prompt is None:
                continue

            target_value, rows, cols = get_value(row, table, metadata)
            if target_value is None or target_value == "":
                continue

            new_sample = [row["pdf name"], row["page nbr"], row["table nbr"],
                          row["gri"], prompt, metadata["operation"],
                          target_value, rows, cols, metadata]
            new_dataset.append(new_sample)
            op_already_seen[j].append(metadata["operation"])

    new_df = pd.DataFrame(new_dataset, columns=["pdf name", "page nbr", "table nbr", "gri", "question", "question_type_ext", "value", "row indices", "col indices", "metadata"])
    new_df.to_csv("gri-qa_multistep.csv", index=False)
    print(len(new_df))