import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
import seaborn as sns



# Absolute or relative path to the folder containing the file.
df = pd.read_csv(os.path.join(os.getcwd(), "star_classification.csv"), na_values=['NA', '?'])

path = "C:/Users/alysh/ANACONDASTUFF/Report/Spacedata" 

# Amount of features + samples
row_count = len(df.axes[0])
cols_count = len(df.axes[1])
# print(f'The DataFrame has {row_count} rows.')
# print(f'The DataFrame has {cols_count} columns.')

# Statistical summary
#numeric_summary = df.describe()

#print("Statistical Summary:")
#print(numeric_summary)

# Checking for missing and null values

# Drop any missing values
# print(df.isnull().any())
# dropna = df[' '].dropna()
# df[' '] = df[' '].fillna(dropna)
# print(df.isnull().any())

# print(df.tail)
# print(df.head)

# Seeing how redshift impacts class outcome
x = df[['redshift']]
y = df['class']

# Converting strings to numerical 
label_encoder = LabelEncoder()
# y_numeric = label_encoder.fit_transform(y)
label_encoder.fit_transform(y)

# SMOTE function and resampling the data
smote = SMOTE(random_state=42)
print('Original data %s' % Counter(y))
x, y = smote.fit_resample(x, y)
print('Resampled data %s' % Counter(y))
x_resampled, y_resampled = smote.fit_resample(x, y)

# Percentage of each class based on the resampled data
class_resampled = pd.Series(y).value_counts()
records_resamples = len(y)

# Df with the resampled data
df_resampled = pd.DataFrame(x_resampled, columns=['redshift'])
df_resampled['class'] = y_resampled

# The bar chart based on the resampled data
class_resampled.plot(kind='bar', color='cornflowerblue', edgecolor='black')

def plt_property(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    
plt.grid(axis='both', linestyle='--', alpha=0.7)
plt_property('Percentage of Each Class', 'Percentage', 'Class')
plt.show()

# Simplified scatter plot for 'redshift' and 'class' to show impact
plt.figure(figsize=(12, 8))
plt_property('Scatter plot of Class vs. Redshift', 'Class', 'Redshift')
sns.scatterplot(data=df_resampled, x='class', y='redshift', palette='Blues', marker='o', alpha=0.8)
plt.xticks(range(len(label_encoder.classes_)), label_encoder.classes_)
plt.show()

# Robust feature scaling as opposed to Z-normalisation (was more efficient)
numerical_features = df.select_dtypes(include=['float64', 'int64']).columns

# Robust scaling using RobustScaler
scaling = RobustScaler()
df[numerical_features] = scaling.fit_transform(df[numerical_features])

# First few rows of the scaled df
print(df.head())

