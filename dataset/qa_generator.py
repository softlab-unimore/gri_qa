import inspect
import math
import os
import pickle
import pandas as pd
import json
import csv
import random
import ast

from copy import deepcopy
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from functools import lru_cache, partial

from tqdm import tqdm

load_dotenv()

@lru_cache()
def extract_table(pdf_name, page_nbr, table_nbr):
    file_name = f"annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv"

    try:
        table = pd.read_csv(file_name, sep=";",
                            quoting=csv.QUOTE_NONE, escapechar='\\')
    except:
        print(
            f"Error with annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv")
        table = None
    return table

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


class OpenAIModel:
    def __init__(self, model_name="gpt-4o-mini", temperature=0.1, max_retries=2, max_tokens=None, timeout=None):
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            max_retries=max_retries,
        )
        
        self.messages = []
        
    def reset_messages(self):
        self.messages = []
        
    def set_system_message(self, system_message):
        self.reset_messages()
        self.messages.append(["system", system_message])
        
    def invoke(self, prompt):
        messages = deepcopy(self.messages)
        messages.append(["human", prompt])
        return self.llm.invoke(messages).content


class QuestionGenerator:
    def __init__(self):
        self.fn_descriptions = self.get_fn_descriptions()
        self.llm = OpenAIModel()
        system_message = """You are a helpful assistant who excels at generating questions.
        You always generate questions that relate to the answer and that are true. You never make things up.
        Do not write anything else other than the requested question. Do not use any Markdown formatting."""
        self.llm.set_system_message(system_message)
        
        self.prompt = """Create a question grounded on the table that can be resolved by the answer.
        The answer has been obtained by applying a function on the table. So, to create the question, strictly follow the function description and use that information to create your question.

        # TABLE
        
        {}
        
        # FUNCTION DESCRIPTION
        
        {}
        
        # ANSWER
        
        {}
        
        # QUESTION
        
        
        """
        
    def get_fn_descriptions(self):
        rg_methods, qrg_methods, rrg_methods = set(dir(ResponseGenerator)), \
                                               set(dir(QuantitativeResponseGenerator)), \
                                               set(dir(RelationResponseGenerator))

        qrg_methods = qrg_methods.difference(rg_methods)
        rrg_methods = rrg_methods.difference(rg_methods)
        
        return {
            method: getattr(generator_class, method).__doc__
            for methods, generator_class in [
                (qrg_methods, QuantitativeResponseGenerator),
                (rrg_methods, RelationResponseGenerator)
            ]
            for method in methods
        }
        
    def generate_question(self, row):
        
        table = extract_table(str(row["pdf name"]), str(row["page nbr"]), str(row["table nbr"]))
        answer = str(row["value"])
        function_description = self.fn_descriptions[row["question_type_ext"]]
        fn_details = row["fn_details"]
        
        if row["question_type_ext"] == "rank":
            function_description += f"\n\nIn this case, firstk is equal to {int(row['firstk'])}."
            
            if "desc" in fn_details["keywords"].keys():
                word = fn_details["keywords"]["desc"]
            else:
                for dictionary in fn_details["args"]:
                    if dictionary["name"] == "desc":
                        word = dictionary["default"]
                        break
            
            function_description += f"Also, desc is equal to {word}.\nThe row and column indices whose values have been considered for the ranking are:\n"
            for row_idx, col_idx in zip(row["row indices"], row["col indices"]):
                function_description += f"- Row \"{table.iloc[row_idx-2,0]}\" with column \"{table.columns[col_idx-1]}\"\n"
            #function_description += "Question template: \"What are the `firstk` values \""
            
        elif row["question_type_ext"] == "comparative" or row["question_type_ext"] == "superlative":
            if "maximise" in fn_details["keywords"].keys():
                word = fn_details["keywords"]["maximise"]
            else:
                for dictionary in fn_details["args"]:
                    if dictionary["name"] == "maximise":
                        word = dictionary["default"]
                        break

            function_description += f"\n\nIn this case, maximise is equal to {word}.\nThe row and column indices whose values have been considered for the comparison are:\n"
            for row_idx, col_idx in zip(row["row indices"], row["col indices"]):
                function_description += f"- Row \"{table.iloc[row_idx,0]}\" with column \"{table.columns[col_idx]}\"\n"
        else:
            function_description += f"\n\nThe row and column indices whose values have been considered for the operation are:\n"
            for row_idx, col_idx in zip(row["row indices"], row["col indices"]):
                function_description += f"- Row \"{table.iloc[row_idx,0]}\" with column \"{table.columns[col_idx]}\"\n"
        
        full_prompt = self.prompt.format(table, function_description, answer)
        print(row)
        res = self.llm.invoke(full_prompt)
        print(res)
        return res

    
    def generate_questions(self, df):
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df[1:], columns=df[0])
            except:
                raise ValueError(f"can't generate questions from a non pd.DataFrame object")
    
        for i, row in df.iterrows():
            df.at[i,"question"] = self.generate_question(row)
        return df
    

class ResponseGenerator:
    def __init__(self, df):
        self.df = self.clean_df(df)
        self.units_of_measure = self.get_units_of_measure("units_of_measure.txt")
        
    def find_unit_of_measure(self, row):
        row = ''.join(row.astype(str)).replace(' ','').lower().strip() #join the whole row as a unique string
        for um in self.units_of_measure:
            um = um.strip()
            if um in row:
                return um
        return None
        
    def get_units_of_measure(self, file_path):
        units_of_measure = []
        
        with open(file_path, 'r') as file:
            for line in file:
                line = line.replace(" ","").lower()
                units_of_measure.append(line)
        
        return units_of_measure
    
    def get_acceptable_rows(self, table, row_idx):
        if (base_unit_of_measure := self.find_unit_of_measure(table.iloc[row_idx,:])) is None:
            return []

        row_indices = []
        for i in range(len(table)):
            if i == row_idx:
                continue
                    
            if self.find_unit_of_measure(table.iloc[i,:]) == base_unit_of_measure:
                row_indices.append(i)
        return row_indices

    def clean_df(self, df):
        df["Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)"] = df["Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)"].fillna(0.0)

        df = df[df["Error (0 no error, 1 value err, 2 unrelated, 3 hierarchical)"].isin([0.0,1.0,3.0])]
        df = df[df["row"].notna()]
        df = df[df["column"].notna()]
        
        return df

    def is_year(self, num):
        try:
            num = int(float(num))
            if len(str(num)) == 4:
                if 1900 <= num <= 2100:
                    return True
        except:
            pass
            
        return False

    def is_table_row_correct(self, table, row_idx):
        correct = 0
        row = table.iloc[row_idx, :]
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
        return False

    def is_table_column_correct(self, table, col_idx, row_idx):
        correct = 0
        col = table.columns[col_idx]

        if self.is_year(col) and len(self.get_acceptable_rows(table, row_idx)) >= 1:
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
        return False

    def inspect_function(self, fn):
        if isinstance(fn, partial):
            # Extract details from the partial function
            sig = inspect.signature(fn.func)
            defaults = {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}
            details = {
                'type': 'partial',
                'name': fn.func.__name__,
                'args': fn.args,
                'keywords': {**defaults, **fn.keywords}  # Merge defaults with the partial's keywords
            }
        else:
            # Extract details for a normal function
            sig = inspect.signature(fn)
            details = {
                'type': 'function',
                'name': fn.__name__,
                'args': [
                    {
                        'name': param.name,
                        'default': param.default if param.default is not inspect.Parameter.empty else None
                    }
                    for param in sig.parameters.values()
                ],
                "keywords": {}
            }
        return details


    def generate(self):
        new_dataset = deepcopy(self.dataset_schema)
        row_column_qa = 0  # 0 for row-based queries, 1 for column-based queries

        def process_table(pdf_name, page_nbr, table_nbr, row, col):
            """
            extract and validate a table based on row or column orientation
            """

            table = extract_table(pdf_name, page_nbr, table_nbr)
            if table is None:
                return None, False

            validate_fn = partial(self.is_table_row_correct, row_idx=row) if not row_column_qa \
                          else partial(self.is_table_column_correct, col_idx=col, row_idx=row)
            table_correctness = validate_fn(table)
            return table, table_correctness

        for i, row in tqdm(self.df.iterrows()):
            pdf_name = str(row["pdf name"])
            page_nbr = str(row["page nbr"])
            table_nbr = str(row["table nbr"])
            gri = str(row["gri"])
            col = int(float(row["column"]))-1
            row = int(float(row["row"]))-2
            
            table, table_correctness = process_table(pdf_name, page_nbr, table_nbr, row, col)
            if table is None or not table_correctness:
                continue
            if to_float(table.iloc[row,col]) is None:
                continue

            fn_idx = i % len(self.fns)
            try:
                question_type_ext = self.fns[fn_idx].__name__
            except:
                question_type_ext = self.fns[fn_idx].func.__name__ #accounting for functools.partial functions
                
            result, question, row_indices, col_indices, firstk = self.create_sample(
                table, fn_idx, row_column_qa, row, col)
            
            if result is None:
                continue
            if question_type_ext == "superlative" and len(row_indices) <= 2:
                question_type_ext = "comparative" #binary superlative questions are comparative questions

            new_row = [
                pdf_name, gri, page_nbr, table_nbr,
                question, self.__class__.__name__, question_type_ext, result,
                row_indices, col_indices, row_column_qa, firstk, self.inspect_function(self.fns[fn_idx])
            ]
            new_dataset.append(new_row)
            
            if i != 0 and not (i+1) % len(self.fns):
                row_column_qa = 1-row_column_qa

        return new_dataset

    def create_sample(self, table, question_idx, row_column_qa, row_idx, col_idx):
        result, question = None, None

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
            try:
                correct, indices = zip(
                    *[(converted, i) for i, value in enumerate(data) if (converted := to_float(value)) is not None]
                )
            except:
                correct, indices = [], []

            return list(correct), list(indices)

        def process_values_based_on_mode():
            """
            process values based on the row/column mode and return the result
            """

            if not row_column_qa:
                col_idx2 = get_random_index(
                    len(table.columns), exclude=col_idx, col_names=table.columns)
                value2 = to_float(table.iloc[row_idx, col_idx2])
                chosen_row_idx = row_idx
                chosen_col_idx = col_idx2
            else:
                row_idx2 = get_random_index(len(table), exclude=row_idx)
                value2 = to_float(table.iloc[row_idx2, col_idx])
                chosen_row_idx = row_idx2
                chosen_col_idx = col_idx
                
            return value2, chosen_row_idx, chosen_col_idx

        def create_question_from_n_values():
            if not row_column_qa:
                col_indices, selected_columns = zip(*[(i, col) for i, col in enumerate(table.columns) if self.is_year(col)])
                col_indices, selected_columns = list(col_indices), list(selected_columns)
                row_indices = [row_idx for _ in col_indices]
                
                iterable = table.loc[row_idx, selected_columns]
            else:
                #unit of measure of the initial row
                row_indices = self.get_acceptable_rows(table, row_idx)
                col_indices = [col_idx for _ in row_indices]
                iterable = table.iloc[row_indices, col_idx]

            numeric_values, indices = get_numeric_values(iterable)
            try:
                row_indices, col_indices = zip(*[(row_indices[i], col_indices[i]) for i in range(len(row_indices)) if i in indices])
                row_indices, col_indices = list(row_indices), list(col_indices)
            except:
                row_indices, col_indices = [], []
                
            if len(numeric_values) < 1:
                return None, None, None, None
            
            if function_name == "rank":
                firstk = random.randint(2, len(numeric_values)+1)
                return self.fns[question_idx](numeric_values, firstk=firstk), row_indices, col_indices, firstk
            
            return self.fns[question_idx](numeric_values), row_indices, col_indices, None
        
        def create_question_from_two_values():
            value = to_float(table.iloc[row_idx, col_idx])
            value2, row_idx2, col_idx2 = process_values_based_on_mode()
            if value is None or value2 is None:
                return None, None, None, None

            return self.fns[question_idx]([value, value2]), [row_idx, row_idx2], [col_idx, col_idx2], None

        while result is None:
            try:
                function_name = self.fns[question_idx].__name__
            except:
                function_name = self.fns[question_idx].func.__name__ #accounting for functools.partial functions
            
            if function_name in ["reduction_percentage", "increase_percentage"] and to_float(table.iloc[row_idx, col_idx]) == 0:
                result = question = row_indices = col_indices = firstk = None
                break #can't calculate the percentage increase/reduction of a zero initial value
            
            if function_name in ["rank", "superlative"]:
                result, row_indices, col_indices, firstk = create_question_from_n_values()
            elif function_name in ["sum", "average"]:
                two_or_more = random.randint(0,1) #0 to create binary questions, 1 otherwise
                if not two_or_more:
                    result, row_indices, col_indices, firstk = create_question_from_two_values()
                else:
                    result, row_indices, col_indices, firstk = create_question_from_n_values()
            else:
                result, row_indices, col_indices, firstk = create_question_from_two_values()

        return result, question, row_indices, col_indices, firstk


class QuantitativeResponseGenerator(ResponseGenerator):
    def __init__(self, df):
        super(QuantitativeResponseGenerator, self).__init__(df)
        self.fns = [
            self.average,
            self.sum,
            self.reduction_difference,
            self.reduction_percentage,
            self.increase_difference,
            self.increase_percentage
        ]
        
        random.shuffle(self.fns)
        
        self.dataset_schema = [["pdf name", "gri", "page nbr", "table nbr",
                                "question", "question_type", "question_type_ext", "value",
                                "row indices", "col indices", "row/column spanning", "firstk", "fn_details"]]

    def average(self, values, to_str=True):
        """
        Calculate the average of a list of numeric values.

        Args:
            values (list of float): The list of numeric values to average.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The average of the values, as a string if `to_str` is True, otherwise as a float.
        """
        
        res = round(float(sum(values)) / len(values),2)
        #print(f"Average: {res}")
        return str(res) if to_str else res

    def sum(self, values, to_str=True):
        """
        Calculate the sum of a list of numeric values.

        Args:
            values (list of float): The list of numeric values to sum.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The sum of the values, as a string if `to_str` is True, otherwise as a float.
        """
        
        res = round(sum(values),2)
        #print(f"Sum: {res}")
        return str(res) if to_str else res

    def reduction_percentage(self, values, to_str=True):
        """
        Calculate the reduction percentage between the first and second numerical values.

        Args:
            values (list of float): A list containing exactly two numeric values, where the first is the initial value, 
                                    and the second is the reduced value.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The reduction percentage, as a string if `to_str` is True, otherwise as a float.

        Raises:
            ValueError: If the list `values` does not contain exactly two elements.
            
        E.g.:
            if values[0] is 150 and values[1] is 100, the percentage reduction is 33.33%.
            if values[0] is 100 and values[1] is 150, the percentage reduction is -33.33%.
        """

        if len(values) != 2:
            raise ValueError(
                f"Can't calculate the reduction percentage of more than 2 values")

        res = round((values[0] - values[1]) * 100 / values[0],2)
        #print(f"Reduction percentage: {res}")
        return str(res) if to_str else res

    def reduction_difference(self, values, to_str=True):
        """
        Calculate the reduction raw difference between the first and second numerical values.

        Args:
            values (list of float): A list containing exactly two numeric values, where the first is the initial value, 
                                    and the second is the reduced value.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The reduction difference, as a string if `to_str` is True, otherwise as a float.

        Raises:
            ValueError: If the list `values` does not contain exactly two elements.
            
        E.g.:
            if values[0] is 150 and values[1] is 100, the reduction is 50.
            if values[0] is 100 and values[1] is 150, the reduction is -50.
        """

        if len(values) != 2:
            raise ValueError(
                f"Can't calculate the reduction difference of more than 2 values")

        res = round(values[0] - values[1], 2)
        #print(f"Reduction difference: {res}")

        return str(res) if to_str else res

    def increase_percentage(self, values, to_str=True):
        """
        Calculate the increase percentage between the first and second numerical values.

        Args:
            values (list of float): A list containing exactly two numeric values, where the first is the original value, 
                                    and the second is the increased value.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The increase percentage, as a string if `to_str` is True, otherwise as a float.
        
        E.g.:
            if values[0] is 150 and values[1] is 100, the percentage increase is -33.33%
            if values[0] is 100 and values[1] is 150, the percentage increase is 33.33%
        """
        
        res = -self.reduction_percentage(values, to_str=False)
        return str(res) if to_str else res

    def increase_difference(self, values, to_str=True):
        """
        Calculate the increase difference between two numeric values.

        Args:
            values (list of float): A list containing exactly two numeric values, where the first is the original value, 
                                    and the second is the increased value.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The increase difference, as a string if `to_str` is True, otherwise as a float.
            
        E.g.:
            if values[0] is 150 and values[1] is 100, the increase is -50
            if values[0] is 100 and values[1] is 150, the increase is 50
        """
        
        res = -self.reduction_difference(values, to_str=False)
        return str(res) if to_str else res


class RelationResponseGenerator(ResponseGenerator):
    def __init__(self, df):
        super(RelationResponseGenerator, self).__init__(df)
        self.fns = [
            self.rank,
            partial(self.rank, desc=True),
            self.superlative,
            partial(self.superlative, maximise=False),
            self.comparative,
            partial(self.comparative, maximise=False)
        ]
        
        random.shuffle(self.fns)

        self.dataset_schema = [["pdf name", "gri", "page nbr", "table nbr",
                                "question", "question_type", "question_type_ext", "value",
                                "row indices", "col indices", "row/column spanning", "firstk", "fn_details"]]

    def rank(self, values, firstk, desc=False):
        """
        Rank the values and return the `firstk` values.

        Args:
            values (list of float): The list of numeric values to rank.
            firstk (int): The number of initial values to return.
            desc (bool, optional): If True, sort in descending order, otherwise ascending. Defaults to False.

        Returns:
            str: A comma-separated string of the initial `firstk` values in the specified order.
        """
        
        return ', '.join([str(value) for value in sorted(values, reverse=desc)][:firstk])

    def superlative(self, values, maximise=True, to_str=True):
        """
        Return the maximum or minimum value from a list of values.

        Args:
            values (list of float): The list of numeric values to evaluate.
            maximise (bool, optional): If True, return the maximum value, otherwise the minimum. Defaults to True.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The maximum or minimum value, as a string if `to_str` is True, otherwise as a float.
        """

        fn = max if maximise else min
        return str(fn(values)) if to_str else fn(values)

    def comparative(self, values, maximise=True, to_str=True):
        """
        Compare two values and return the maximum or minimum value.

        Args:
            values (list of float): A list containing exactly two numeric values to compare.
            maximise (bool, optional): If True, return the maximum value, otherwise the minimum. Defaults to True.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The maximum or minimum value, as a string if `to_str` is True, otherwise as a float.

        Raises:
            ValueError: If the list `values` does not contain exactly two elements.
        """
        
        if len(values) != 2:
            raise ValueError(
                f"Can't calculate the comparison between more than 2 values")

        return self.superlative(values, maximise, to_str)


class KeywordResponseGenerator(ResponseGenerator):
    def __init__(self, df):
        super(KeywordResponseGenerator, self).__init__(df)
        self.llm = OpenAIModel()
        system_message = "You are a helpful assistant that assists people in changing questions by using different keywords. Write exclusively the requested question. Do not use any Markdown formatting."
        self.llm.set_system_message(system_message)
        self.prompt = """You will be provided with a table (in HTML), along with a question and its corresponding answer, both derived from the information in the table.
        Your task is to rephrase the original question using different keywords while ensuring the following: the rephrased question retains the same meaning as the original, remains unambiguous, and is still answerable by the given answer based on the table's context.

        # TABLE
        
        {}
        
        # QUESTION
        
        {}
        
        # ANSWER
        
        {}
        
        # NEW QUESTION

        """
        
    def generate(self, path):  # gri-qa_extr.csv
        file_type = path.split(".")[-1]
        if file_type == "tsv":
            sep = "\t"
        elif file_type == "csv":
            sep = ","
        else:
            raise TypeError(
                f"{path} has the wrong file type. The file must be .tsv or .csv")

        df = pd.read_csv(path, sep=sep)
        new_dataset = [df.columns]

        for i, row in tqdm(df.iterrows()):
            message_dp = deepcopy(self.prompt)

            gri = str(row["gri"])
            page_nbr = str(row["page nbr"])
            pdf_name = str(row["pdf name"])
            table_nbr = str(row["table nbr"])
            question = str(row["question"])
            answer = str(row["value"])
            
            file_name = f"annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv"

            try:
                table = pd.read_csv(file_name, sep=";", escapechar='\\').to_html(index=False)
            except:
                print(
                    f"Error with annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv")
                continue

            try:
                ai_msg = self.llm.invoke(message_dp.format(table, question, answer))
                #print(f"{question} --- {ai_msg}")
                result = ai_msg
            except:
                print(
                    f"Malformed node for annotation/{pdf_name.split('.')[0].strip()}/{page_nbr}_{table_nbr}.csv")
                continue

            new_row = deepcopy(row)
            new_row["question"] = result
            new_dataset.append(new_row.tolist())
            if i % 100 == 0:
                with open(f'checkpoint{i}.pkl', 'wb') as f:
                    pickle.dump(new_dataset, f)

        new_df = pd.DataFrame(new_dataset)
        return new_df 
        

class ExtractiveResponseGenerator(ResponseGenerator):
    def __init__(self, df):
        super(ExtractiveResponseGenerator, self).__init__(df)
        self.dataset_schema = [
            ["pdf name", "gri", "page nbr", "table nbr", "question", "value"]]

        self.llm = OpenAIModel()
        self.llm.set_system_message("You are a helpful assistant that assists people in generating question answer pairs. You never generate question answer pairs that are not known based on the context or that are false.")

        self.prompt = """You will be given a table (in HTML) and some indicators.
        Extract the values that can reply the given indicators. Then, for each value, generate an unambiguous question that can be replied by the value.
        By unambiguous question I mean that, based on the table, the question can have only one answer.
        In particular, the question must ask aggregation operations like summation, difference or percentage increase/decrease of values across different columns (e.g. across different years).
        For example, you can ask "What's the total amount of X across the years?" or "What's the difference of X between YEAR1 and YEAR2?" or "How much has X in YEAR1 decreased in percentage with reference to YEAR2?" or "Is the value of X greater in YEAR1 or YEAR2? YEAR1 / YEAR2". Use different variations of these examples.
        As output, provide a Python dictionary that has the questions as keys and the extracted values are the values of the dictionary. Do not write anything else. Do not provide any Markdown formatting.

        Table: {}
        Topics: {}
        """

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

            ai_msg = self.llm.invoke(message_dp.format(table, indicator_values))

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

if __name__ == "__main__":
    df = pd.read_csv("one-table/gri-qa_extra.csv")
    path_to_dataset = "one-table/gri-qa_kw.csv"
    
    """q_responsegenerator = RelationResponseGenerator(df)
    res = q_responsegenerator.generate()
    
    new_df = pd.DataFrame(res[1:], columns=res[0])

    qg = QuestionGenerator()
    new_df = qg.generate_questions(new_df)
    
    if os.path.exists(path_to_dataset):
        df = pd.read_csv(path_to_dataset)
        new_df = pd.concat([df, new_df], ignore_index=True)
    
    new_df.loc[new_df.drop("question",axis="columns").astype(str).drop_duplicates().index]
    new_df.to_csv(path_to_dataset, index=False)
    print(len(new_df))"""
    
    k_responsegenerator = KeywordResponseGenerator(df)
    new_df = k_responsegenerator.generate("gri-qa_extra.csv")
    new_df.to_csv(path_to_dataset, index=False)
    
    
