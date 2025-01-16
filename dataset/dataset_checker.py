import csv
from ast import literal_eval
import pandas as pd
from functools import lru_cache

from tqdm import tqdm


class Checker:
    def __init__(self):
        pass

    @lru_cache(maxsize=None)
    def read_table(self, path):
        return pd.read_csv(path, sep=";")

    @staticmethod
    def extract_value(df, row_idx, col_idx):
        try:
            return float(df.iloc[row_idx, col_idx])
        except:
            #print(df)
            #print(row_idx, col_idx)
            return df.iloc[row_idx, col_idx]

    def extract_values(self, path, rows, cols):
        df = self.read_table(path)
        try:
            rows = literal_eval(rows)
            cols = literal_eval(cols)
        except:
            rows, cols = [rows], [cols]
        values = []
        for row, col in zip(rows, cols):
            value = self.extract_value(df, int(float(row))-2, int(float(col))-1)
            values.append(value)

        return values

class RelChecker(Checker):
    def __init__(self):
        super(RelChecker, self).__init__()

    @staticmethod
    def rank(values, firstk, desc=False, lowest_highest=False):
        """
        Rank the values and return the `firstk` values.

        Args:
            values (list of float): The list of numeric values to rank.
            firstk (int): The number of initial values to return.
            desc (bool, optional): If True, sort in descending order, otherwise ascending. Defaults to False.

        Returns:
            str: A comma-separated string of the initial `firstk` values in the specified order.
        """
        values = sorted(values)
        if lowest_highest:
            values = values[-firstk:]
        else:
            values = values[:firstk]

        values = [str(round(float(value),2)) for value in sorted(values, reverse=desc)]

        return ', '.join(values)

    @staticmethod
    def superlative(values, maximise=True, to_str=True):
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
        return str(round(float(fn(values)),2)) if to_str else fn(values)

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

        assert (isinstance(values[0], float) or isinstance(values[0], int)) and \
               (isinstance(values[1], float) or isinstance(values[1], int))
        result = values[0] > values[1] if maximise else values[0] < values[1]
        return "yes" if result else "no"


class QuantChecker(Checker):
    def __init__(self):
        super(QuantChecker, self).__init__()

    @staticmethod
    def average(values, to_str=True):
        """
        Calculate the average of a list of numeric values.

        Args:
            values (list of float): The list of numeric values to average.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The average of the values, as a string if `to_str` is True, otherwise as a float.
        """

        res = round(float(sum(values)) / len(values), 2)
        return str(res) if to_str else res

    @staticmethod
    def sum(values, to_str=True):
        """
        Calculate the sum of a list of numeric values.

        Args:
            values (list of float): The list of numeric values to sum.
            to_str (bool, optional): If True, return the result as a string. Defaults to True.

        Returns:
            str or float: The sum of the values, as a string if `to_str` is True, otherwise as a float.
        """

        res = round(sum(values), 2)
        # print(f"Sum: {res}")
        return str(res) if to_str else res

    @staticmethod
    def reduction_percentage(values, to_str=True):
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

        res = round((values[0] - values[1]) * 100 / values[0], 2)
        # print(f"Reduction percentage: {res}")
        return str(res) if to_str else res

    @staticmethod
    def reduction_difference(values, to_str=True):
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
        # print(f"Reduction difference: {res}")

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

class ExtraChecker(Checker):
    def __init__(self):
        super(ExtraChecker, self).__init__()


class CheckerFactory:
    def __init__(self, dataset_path):
        self.rel_checker = RelChecker()
        self.quant_checker = QuantChecker()
        self.extra_checker = ExtraChecker()
        self.dataset_path = dataset_path

    def run(self):
        dataset_df = pd.read_csv(self.dataset_path)
        results = []
        results2 = []
        for i,row in tqdm(dataset_df.iterrows()):
            dataset_abbrv = self.dataset_path.split("/")[-1].split(".")[0].split("_")[-1]
            table_path = f"annotation/{eval(row['pdf name'])[0].split('.')[0]}/{eval(row['page nbr'])[0]}_{eval(row['table nbr'])[0]}.csv"
            if dataset_abbrv in ["rel", "quant"]:
                values = self.rel_checker.extract_values(table_path, row["row indices"], row["col indices"])
            else:
                values = self.extra_checker.extract_values(table_path, row["row"], row["column"])[0]
                values = str(values)
            if values is None:
                results.append("TO BE CHECKED MANUALLY")
                results2.append("TO BE CHECKED MANUALLY")
                continue
            if dataset_abbrv == "rel":
                row["metadata"] = literal_eval(row["metadata"])
                if row["question_type_ext"] == "rank":
                    results.append(self.rel_checker.rank(
                        values,
                        int(row["metadata"]["firstk"]),
                        row["metadata"]["desc"],
                        row["metadata"]["lowest-highest"]
                    ))
                elif row["question_type_ext"] == "comparative":
                    results.append(self.rel_checker.comparative(values, maximise=row["metadata"]["maximise"]))
                elif row["question_type_ext"] == "superlative":
                    results.append(self.rel_checker.superlative(values, maximise=row["metadata"]["maximise"]))
                else:
                    raise ValueError(f"Unknown question type {row['question_type_ext']} for dataset {dataset_abbrv} and row {i} in {self.dataset_path}")

            elif dataset_abbrv == "quant":
                if row["question_type_ext"] == "average":
                    results.append(self.quant_checker.average(values))
                    results2.append(self.quant_checker.average(values[::-1]))
                elif row["question_type_ext"] == "sum":
                    results.append(self.quant_checker.sum(values))
                    results2.append(self.quant_checker.sum(values[::-1]))
                elif row["question_type_ext"] == "reduction_percentage":
                    results.append(self.quant_checker.reduction_percentage(values))
                    results2.append(self.quant_checker.reduction_percentage(values[::-1]))
                elif row["question_type_ext"] == "reduction_difference":
                    results.append(self.quant_checker.reduction_difference(values))
                    results2.append(self.quant_checker.reduction_difference(values[::-1]))
                elif row["question_type_ext"] == "increase_percentage":
                    results.append(self.quant_checker.increase_percentage(values))
                    results2.append(self.quant_checker.increase_percentage(values[::-1]))
                elif row["question_type_ext"] == "increase_difference":
                    results.append(self.quant_checker.increase_difference(values))
                    results2.append(self.quant_checker.increase_difference(values[::-1]))
                elif row["question_type_ext"] == "multi-step":
                    results.append("TO BE CHECKED MANUALLY")
                    results2.append("TO BE CHECKED MANUALLY")
                else:
                    raise ValueError(f"Unknown question type {row['question_type_ext']} for dataset {dataset_abbrv} and row {i} in {self.dataset_path}")
            elif dataset_abbrv == "extra":
                results.append(values)

        dataset_df["automatic check"] = results
        if dataset_abbrv == "quant":
            dataset_df["automatic check 2"] = results2
        return dataset_df

def value_changer_rel(row):
    if row["question_type_ext"] == "comparative":
        return row
    if row["question_type_ext"] == "rank":
        values_str = row["value"].split(", ")
        values = [str(round(float(value),2)) for value in values_str]
        row["value"] = ", ".join(values)
    if row["question_type_ext"] == "superlative":
        row["value"] = str(round(float(row["value"]),2))
    return row

def value_changer_quant(row):
    row["value"] = str(round(float(row["value"].replace(",","")),2))
    return row

def value_changer_extra(row):
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    row["value"] = str(row["value"]).strip().replace("%","")
    row["automatic check"] = str(row["automatic check"]).strip().replace("%","")

    if isfloat(row["value"]):
        if "." in row["value"]:
            row["value"] = row["value"].rstrip("0").rstrip(".")

    if isfloat(row["automatic check"]):
        if "." in row["automatic check"]:
            row["automatic check"] = row["automatic check"].rstrip("0").rstrip(".")

    return row

if __name__ == "__main__":
    dataset_path = "one-table/gri-qa_extra.csv"
    dataset_type = dataset_path.split("/")[-1].split(".")[0].split("_")[-1]
    factory = CheckerFactory(dataset_path)
    df = factory.run()
    if dataset_type == "rel":
        df = df.apply(value_changer_rel, axis="columns")
        df["boolean_check"] = df["value"] == df["automatic check"]
    elif dataset_type == "quant":
        df = df.apply(value_changer_quant, axis="columns")
        df["boolean_check"] = (df["value"] == df["automatic check"]) | (df["value"] == df["automatic check 2"])
    elif dataset_type == "extra":
        df = df.apply(value_changer_extra, axis="columns")
        df["boolean_check"] = df["value"] == df["automatic check"]
    else:
        raise ValueError(f"Unknown dataset type {dataset_type} for dataset {dataset_path}")

    df.to_csv(f"{'/'.join(dataset_path.split('/')[:-1])}/check_{dataset_path.split('/')[-1]}", index=False)
    print(sum(df["boolean_check"]))
    print(len(df))