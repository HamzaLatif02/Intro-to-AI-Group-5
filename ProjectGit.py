import os
import pandas as pd
import matplotlib.pyplot as plt

def plt_property(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Absolute or relative path to the folder containing the file.
df = pd.read_csv(os.path.join(os.getcwd(), "star_classification.csv"), na_values=['NA', '?'])

path = "C:/Users/alysh/ANACONDASTUFF/Report/Spacedata" 

# Amount of features + samples
row_count = len(df.axes[0])
cols_count = len(df.axes[1])
# print(f'The DataFrame has {row_count} rows.')
# print(f'The DataFrame has {cols_count} columns.')


#1.Checking for missing and null values

# Drop any missing values
# print(df.isnull().any())
# dropna = df[' '].dropna()
# df[' '] = df[' '].fillna(dropna)
# print(df.isnull().any())

# print(df.tail)
# print(df.head)