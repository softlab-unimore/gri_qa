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
        return pd.read_csv(path, sep=";", quoting=csv.QUOTE_NONE, escapechar='\\')

    @staticmethod
    def extract_value(df, row_idx, col_idx):
        try:
            return float(df.iloc[row_idx, col_idx])
        except:
            return None

    def extract_values(self, path, rows, cols):
        df = self.read_table(path)
        try:
            rows = literal_eval(rows)
            cols = literal_eval(cols)
        except:
            return None
        values = []
        for row, col in zip(rows, cols):
            value = self.extract_value(df, int(float(row))-2, int(float(col))-1)
            if value is None:
                return None
            values.append(value)

        return values

class RelChecker(Checker):
    def __init__(self):
        super(RelChecker, self).__init__()

    @staticmethod
    def rank(values, firstk, desc=False):
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
            table_path = f"annotation/{row['pdf name'].split('.')[0]}/{row['page nbr']}_{row['table nbr']}.csv"
            values = self.rel_checker.extract_values(table_path, row["row indices"], row["col indices"])
            if values is None:
                results.append("TO BE CHECKED MANUALLY")
                results2.append("TO BE CHECKED MANUALLY")
                continue
            if dataset_abbrv == "rel":
                if row["question_type_ext"] == "rank":
                    print(self.rel_checker.rank(values, int(row["firstk"])))
                elif row["question_type_ext"] == "comparative":
                    print(self.rel_checker.comparative(values, maximise=row["maximise"]))
                elif row["question_type_ext"] == "superlative":
                    print(self.rel_checker.superlative(values, maximise=row["maximise"]))
            elif dataset_abbrv == "quant":
                if row["question_type_ext"] == "average":
                    results.append(self.quant_checker.average(values, to_str=False))
                    results2.append(self.quant_checker.average(values[::-1], to_str=False))
                elif row["question_type_ext"] == "sum":
                    results.append(self.quant_checker.sum(values, to_str=False))
                    results2.append(self.quant_checker.sum(values[::-1], to_str=False))
                elif row["question_type_ext"] == "reduction_percentage":
                    results.append(self.quant_checker.reduction_percentage(values, to_str=False))
                    results2.append(self.quant_checker.reduction_percentage(values[::-1], to_str=False))
                elif row["question_type_ext"] == "reduction_difference":
                    results.append(self.quant_checker.reduction_difference(values, to_str=False))
                    results2.append(self.quant_checker.reduction_difference(values[::-1], to_str=False))
                elif row["question_type_ext"] == "increase_percentage":
                    results.append(self.quant_checker.increase_percentage(values, to_str=False))
                    results2.append(self.quant_checker.increase_percentage(values[::-1], to_str=False))
                elif row["question_type_ext"] == "increase_difference":
                    results.append(self.quant_checker.increase_difference(values, to_str=False))
                    results2.append(self.quant_checker.increase_difference(values[::-1], to_str=False))
                elif row["question_type_ext"] == "multi-step":
                    results.append("TO BE CHECKED MANUALLY")
                    results2.append("TO BE CHECKED MANUALLY")
                else:
                    print(row["question_type_ext"])
            elif dataset_abbrv == "extra":
                pass
        dataset_df["automatic check"] = results
        dataset_df["automatic check 2"] = results2
        return dataset_df

if __name__ == "__main__":
    dataset_path = "one-table/gri-qa_quant.csv"
    factory = CheckerFactory(dataset_path)
    df = factory.run()
    df["boolean_check"] = (df["value"] == df["automatic check"]) | (df["value"] == df["automatic check 2"])
    df.to_csv(f"{'/'.join(dataset_path.split('/')[:-1])}/check_{dataset_path.split('/')[-1]}", index=False)