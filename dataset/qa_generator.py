import math
import pandas as pd
import json
import csv
import random
import ast

from copy import deepcopy
#from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from functools import lru_cache, partial

from tqdm import tqdm

#load_dotenv()

class QAGenerator:
    def __init__(self, df):
        self.df = df

    def is_year(self, num):
        try:
            num = int(float(num))
            if len(str(num)) == 4:
                if 1900 <= num <= 2100:
                    return True
        except:
            pass
            
        return False

    def is_table_row_correct(self, table):
        correct = 0
        for i, row in table.iterrows():
            for j, cell in enumerate(row):
                if not self.is_year(table.columns[j]):
                    continue
                if isinstance(cell, str):
                    cell = cell.replace(",", ".")
                    try:
                        float(cell)
                        if correct:
                            return True
                        else:
                            correct = 1
                    except:
                        pass
            correct = 0
        return False

    def is_table_column_correct(self, table):
        correct = 0
        for col in table.columns:
            if not self.is_year(col):
                continue
            for row_idx in range(len(table)):
                cell = table[col].iloc[row_idx]
                if isinstance(cell, str):
                    cell = cell.replace(",", ".")
                    try:
                        float(cell)
                        if correct:
                            return True
                        else:
                            correct = 1
                    except:
                        pass
            correct = 0
        return False

    @lru_cache()
    def extract_table(self, pdf_name, page_nbr, table_nbr):
        file_name = f"annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv"

        try:
            table = pd.read_csv(file_name, sep=";",
                                quoting=csv.QUOTE_NONE, escapechar='\\')
        except:
            print(
                f"Error with annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv")
            table = None
        return table


    def generate(self):
        new_dataset = deepcopy(self.dataset_schema)
        row_column_qa = 0  # 0 for row-based queries, 1 for column-based queries

        def process_table(pdf_name, page_nbr, table_nbr):
            """
            extract and validate a table based on row or column orientation
            """

            table = self.extract_table(pdf_name, page_nbr, table_nbr)
            if table is None:
                return None, False

            validate_fn = self.is_table_row_correct if not row_column_qa else self.is_table_column_correct
            table_correctness = validate_fn(table)
            return table, table_correctness

        for i, row in tqdm(self.df.iterrows()):
            pdf_name = str(row["pdf name"])
            page_nbr = str(row["page nbr"])
            table_nbr = str(row["table nbr"])
            gri = str(row["gri"])

            table, table_correctness = process_table(pdf_name, page_nbr, table_nbr)
            if table is None or not table_correctness:
                continue

            question_type_ext = i % len(self.fns)
            result, question = self.create_sample(
                table, question_type_ext, row_column_qa)

            new_row = [
                pdf_name, gri, page_nbr, table_nbr,
                question, self.__class__.__name__, question_type_ext, result
            ]
            new_dataset.append(new_row)
            
            if i != 0 and not (i+1) % len(self.fns):
                row_column_qa = 1-row_column_qa

        return new_dataset


    def create_sample(self, table, question_idx, row_column_qa):
        result, question = None, None

        def to_float(value):
            try:
                if isinstance(value, str):
                    value = float(value.replace(",", "."))
                else:
                    value = float(value)
                if math.isnan(value):
                    raise
            except:
                return None
            return value

        def get_random_index(max_index, exclude=None, col_names=[]):
            """
            get a random index, while excluding a specific value (exclude)
            """

            idx = random.randint(0, max_index - 1)
            while (exclude is not None and idx == exclude) or (len(col_names) > 0 and not self.is_year(col_names[idx])):
                idx = random.randint(0, max_index - 1)

            return idx

        def get_numeric_values(data):
            """
            try to convert the values of a list to float by replacing , with . and casting str to float
            """

            return [to_float(value) for value in data if to_float(value) is not None]

        def process_values_based_on_mode():
            """
            process values based on the row/column mode and return the result
            """

            if not row_column_qa:
                col_idx2 = get_random_index(
                    len(table.columns), exclude=col_idx, col_names=table.columns)
                value2 = to_float(table.iloc[row_idx, col_idx2])
            else:
                row_idx2 = get_random_index(len(table), exclude=row_idx)
                value2 = to_float(table.iloc[row_idx2, col_idx])
            return value2

        def create_question_from_n_values():
            iterable = table.loc[
                row_idx, [
                    self.is_year(col) for col in table.columns
                ]
            ] if not row_column_qa else table.iloc[:, col_idx]  # check if the unit measure is the same

            numeric_values = get_numeric_values(iterable)
            if len(numeric_values) < 2:
                return None
            
            if function_name == "rank":
                firstk = random.randint(2, len(numeric_values))
                return self.fns[question_idx](numeric_values, firstk=firstk)
            
            return self.fns[question_idx](numeric_values)
        
        def create_question_from_two_values():
            value = to_float(table.iloc[row_idx, col_idx])
            value2 = process_values_based_on_mode()

            if value is None or value2 is None:
                return None
            if function_name in ["reduction_percentage", "increase_percentage"] and value == 0:
                return None

            return self.fns[question_idx]([value, value2])

        while result is None:
            row_idx = get_random_index(len(table))
            col_idx = get_random_index(len(table.columns))

            try:
                function_name = self.fns[question_idx].__name__
            except:
                function_name = self.fns[question_idx].func.__name__ #accounting for functools.partial functions
                
            if function_name in ["rank", "superlative"]:
                result = create_question_from_n_values()
            elif function_name in ["sum", "average"]:
                two_or_more = random.randint(0,1) #0 to create binary questions, 1 otherwise
                if not two_or_more:
                    result = create_question_from_two_values()
                else:
                    result = create_question_from_n_values()
            else:
                result = create_question_from_two_values()

        return result, question


class QuantitativeQAGenerator(QAGenerator):
    def __init__(self, df):
        super(QuantitativeQAGenerator, self).__init__(df)
        self.df = df
        self.fns = [
            self.average,
            self.sum,
            self.reduction_difference,
            self.reduction_percentage,
            self.increase_difference,
            self.increase_percentage
        ]
        self.dataset_schema = [["pdf name", "gri", "page nbr", "table nbr",
                                "question", "question_type", "question_type_ext", "value"]]

    def average(self, values, to_str=True):
        res = round(float(sum(values)) / len(values),2)
        #print(f"Average: {res}")
        return str(res) if to_str else res

    def sum(self, values, to_str=True):
        res = round(sum(values),2)
        #print(f"Sum: {res}")
        return str(res) if to_str else res

    def reduction_percentage(self, values, to_str=True):
        """
        percentage of values[0] w.r.t. values[1]
        """

        if len(values) != 2:
            raise ValueError(
                f"Can't calculate the reduction percentage of more than 2 values")

        res = round((values[0] - values[1]) * 100 / values[0],2)
        #print(f"Reduction percentage: {res}")
        return str(res) if to_str else res

    def reduction_difference(self, values, to_str=True):
        """
        percentage of values[0] w.r.t. values[1]
        """

        if len(values) != 2:
            raise ValueError(
                f"Can't calculate the reduction difference of more than 2 values")

        res = round(values[0] - values[1], 2)
        #print(f"Reduction difference: {res}")

        return str(res) if to_str else res

    def increase_percentage(self, values, to_str=True):
        res = -self.reduction_percentage(values, to_str=False)
        return str(res) if to_str else res

    def increase_difference(self, values, to_str=True):
        res = -self.reduction_difference(values, to_str=False)
        return str(res) if to_str else res


class RankingQAGenerator(QAGenerator):
    def __init__(self, df):
        super(RankingQAGenerator, self).__init__(df)
        self.df = df
        self.fns = [
            self.rank,
            partial(self.rank, desc=True),
            self.superlative,
            partial(self.superlative, maximise=False),
            self.comparative,
            partial(self.comparative, maximise=False)
        ]
        self.dataset_schema = [
            ["pdf name", "gri", "page nbr", "table nbr", "question", "value"]]

    def rank(self, values, firstk, desc=False):
        return ', '.join([str(value) for value in sorted(values, reverse=desc)][:firstk])

    def superlative(self, values, maximise=True, to_str=True):
        fn = max if maximise else min
        return str(fn(values)) if to_str else fn(values)

    def comparative(self, values, maximise=True, to_str=True):
        if len(values) != 2:
            raise ValueError(
                f"Can't calculate the comparison between more than 2 values")

        return self.superlative(values, maximise, to_str)


class ExtractiveQAGenerator:
    def __init__(self):
        self.dataset_schema = [
            ["pdf name", "gri", "page nbr", "table nbr", "question", "value"]]

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

        self.prompt = """You will be given a table (in HTML) and some indicators.
        Extract the values that can reply the given indicators. Then, for each value, generate an unambiguous question that can be replied by the value.
        By unambiguous question I mean that, based on the table, the question can have only one answer.
        In particular, the question must ask aggregation operations like summation, difference or percentage increase/decrease of values across different columns (e.g. across different years).
        For example, you can ask "What's the total amount of X across the years?" or "What's the difference of X between YEAR1 and YEAR2?" or "How much has X in YEAR1 decreased in percentage with reference to YEAR2?" or "Is the value of X greater in YEAR1 or YEAR2? YEAR1 / YEAR2". Use different variations of these examples.
        As output, provide a Python dictionary that has the questions as keys and the extracted values are the values of the dictionary. Do not write anything else. Do not provide any Markdown formatting.

        Table: {}
        Topics: {}
        """

        self.messages = [
            [
                "system",
                "You are a helpful assistant that assists people in generating question answer pairs. You never generate question answer pairs that are not known based on the context or that are false.",
            ],
            [
                "human",
                ""
            ],
        ]

        self.queries = "json_config/en_queries_extended.json"

    def generate(self, path):  # annotation/data.tsv
        file_type = path.split(".")[-1]
        if file_type == "tsv":
            sep = "\t"
        elif file_type == "csv":
            sep = ","
        else:
            raise TypeError(
                f"{path} has the wrong file type. The file must be .tsv or .csv")

        df = pd.read_csv(path, sep="\t")
        new_dataset = deepcopy(self.dataset_schema)

        with open(self.queries, 'r') as file:
            data = json.load(file)

        for i, row in df.iterrows():
            message_dp = deepcopy(self.prompt)

            gri = str(row["GRI"])
            page_nbr = str(row["page"])
            pdf_name = str(row["pdf_name"])
            table_nbr = str(row["nr_table_in_page"])

            print(
                f"annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv")
            file_name = f"annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv"
            try:
                table = pd.read_csv(file_name, header=None, sep=";",
                                    quoting=csv.QUOTE_NONE, escapechar='\\').to_html(index=False)
            except:
                print(
                    f"Error with annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv")
                continue

            indicator_values = [
                v for k, v in data.items() if gri == k.split("-")[0]]
            indicator_values = '; '.join(indicator_values)

            message_dp = message_dp.format(table, indicator_values)

            self.messages[1][1] = message_dp

            ai_msg = self.llm.invoke(self.messages)

            try:
                dict_values = ast.literal_eval(ai_msg.content)
            except:
                print(
                    f"Malformed node for annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv")
                continue

            for k, v in dict_values.items():
                new_dataset.append([pdf_name, gri, page_nbr, table_nbr, k, v])

        new_df = pd.DataFrame(new_dataset)
        new_df.to_csv("qa_dataset_aggr.csv", sep=";")

      
def to_float(value):
            try:
                value = float(value.replace(",", "."))
            except:
                return None
            return value      

if __name__ == "__main__":
    df = pd.read_csv("qa_dataset.csv")
    q_qagenerator = RankingQAGenerator(df)
    res = q_qagenerator.generate()
    print(res)
    print(len(res))
