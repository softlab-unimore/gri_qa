import pandas as pd
import csv

df = pd.read_csv("311_0.csv", sep=";") #, quoting=csv.QUOTE_NONE, escapechar='\\')

print(df.head())
