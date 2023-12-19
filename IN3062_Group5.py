import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

def plt_property(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Absolute or relative path to the folder containing the file.
df = pd.read_csv("/Users/apple/Desktop/uni/year3/IntroToAI/cw/star_classification.csv", na_values=['NA', '?'])

# Amount of features + samples
row_count = len(df.axes[0])
cols_count = len(df.axes[1])
# print(f'The DataFrame has {row_count} rows.')
# print(f'The DataFrame has {cols_count} columns.')

#renaming the columns to more understandable heading
df = df.rename(columns = {'g': 'Green Light'})
df = df.rename(columns = {'u': 'Ultraviolet Light'})
df = df.rename(columns = {'r': 'Red Light'})
df = df.rename(columns = {'i': 'Infrared Light'})
df = df.rename(columns = {'delta': 'Declination Angle'})
df = df.rename(columns = {'alpha': 'Ascension Angle'})

#1.Checking for missing and null values

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

#Using the SMOTE function and resampling the data
smote = SMOTE(random_state=42)
print('Original data %s' % Counter(y))
x, y = smote.fit_resample(x, y)
print('Resampled data %s' % Counter(y))
x_resampled, y_resampled = smote.fit_resample(x, y)
# Calculate the percentage of each class based on the resa

# Selecting the 'redshift' column for scaling
redshift_data = df[['redshift']]

# Creating a StandardScaler instance
scaler = StandardScaler()

# Fitting and transforming the 'redshift' data
scaled_redshift = scaler.fit_transform(redshift_data)

# Adding the scaled redshift back to the dataframe
df['scaled_redshift'] = scaled_redshift

# Displaying the first few rows of the updated redshift feature
comparison_df = pd.DataFrame({'Original': df['redshift'], 'Scaled': df['scaled_redshift']})
print(comparison_df.head())

print()

# Plotting the scatter plot between 'redshift' and 'class'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='class', y='redshift', palette='viridis')

plt.title('Scatter Plot of Redshift vs Class')
plt.xlabel('Class')
plt.ylabel('Redshift')
plt.show()

print()

# Code to Build a Naive Model and Evaluate It
# Selecting 'redshift' as the feature and 'class' as the target
X = df[['redshift']]
y = df['class']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Calculate size of training dataset
traindata_size = X_train.size + y_train.size


#K-Nearest Neighbour model
def knn_model(nneighbours):
    # Using K-Nearest Neighbors as a naive model
    knn = KNeighborsClassifier(n_neighbors=nneighbours)
    knn.fit(X_train, y_train)

    # Predictions
    y_pred = knn.predict(X_test)

    # Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    
    print('Number of neighbours:', nneighbours)
    print('Baseline Model Accuracy:', accuracy)
    print('Classification Report:\n', class_report)
    print()

#Run the model using different values for number of neighbours to see diferrence in results
knn_model(3)
knn_model(5)
knn_model(10)
knn_model(100)
knn_model(math.isqrt(traindata_size))
knn_model(1000)
