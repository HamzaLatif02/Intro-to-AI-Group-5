import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import math
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def plt_property(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

# Absolute or relative path to the folder containing the file.
df = pd.read_csv("/Users/apple/Desktop/uni/year3/IntroToAI/cw/star_classification.csv", na_values=['NA', '?'])

# Amount of features + samples
row_count = len(df.axes[0])
cols_count = len(df.axes[1])
print(f'The DataFrame has {row_count} rows.')
print(f'The DataFrame has {cols_count} columns.')

#renaming the columns to more understandable heading
df = df.rename(columns = {'g': 'Green Light'})
df = df.rename(columns = {'u': 'Ultraviolet Light'})
df = df.rename(columns = {'r': 'Red Light'})
df = df.rename(columns = {'i': 'Infrared Light'})
df = df.rename(columns = {'delta': 'Declination Angle'})
df = df.rename(columns = {'alpha': 'Ascension Angle'})

#1.Checking for missing and null values

# Drop any missing values
print(df.isnull().any())
dropna = df[' '].dropna()
df[' '] = df[' '].fillna(dropna)
print(df.isnull().any())

print(df.tail)
print(df.head)

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
sns.scatterplot(data=df, x='class', y='scaled_redshift', palette='viridis')

plt.title('Scatter Plot of Redshift vs Class')
plt.xlabel('Class')
plt.ylabel('Scaled_Redshift')
plt.show()

print()

# Code to Build a Naive Model and Evaluate It
# Selecting 'redshift' as the feature and 'class' as the target
X = df[['scaled_redshift']]
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

# Run the model using different values for number of neighbours to see diferrence in results
knn_model(3)
knn_model(5)
knn_model(10)
knn_model(100)
knn_model(math.isqrt(traindata_size))
knn_model(1000)

# Code to build a Neural Network model and evaluate it
# Select all relevant features as the features and 'class' as target
X = df.drop(['class', 'obj_ID', 'run_ID', 'rerun_ID','MJD', 'redshift'], axis=1)

# Assign each class (Galaxy, Star, QSO) to an integer value, the one hot encode. 
label_encoder = LabelEncoder()
df['class'] = label_encoder.fit_transform(df['class'])
y = keras.utils.to_categorical(df['class'])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural Network model
def nn_model(n_hidden_layers, hidden_activation, n_hidden_units, n_epochs):
    print(f'Number of hidden layers: {n_hidden_layers}, Hidden layers activation type: {hidden_activation}')
    print(f'Number of hidden units: {n_hidden_units}, Number of epochs: {n_epochs}')
    
    i = 1
    model = Sequential()
    model.add(Dense(n_hidden_units, activation=hidden_activation, input_shape=(X_train.shape[1],)))
    # Add hidden layers
    while i < n_hidden_layers:
        model.add(Dense(n_hidden_units, activation=hidden_activation))
        i+=1
    
    model.add(Dense(y.shape[1],activation='softmax'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train,verbose=2, epochs=n_epochs, batch_size=32)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy}')
    print()

# Run the model using different parameters to see difference in results
nn_model(1,'relu',64,10)
nn_model(1,'relu',64,200)
nn_model(1,'relu',1024,200)
nn_model(2,'relu',1024,200)

# For the last model, transform the data so it is normally distributed
sc = StandardScaler()
sc.fit(X_train)
X_train= sc.transform(X_train)
X_test = sc.transform(X_test)

nn_model(2, 'relu', 64, 200)


# Function to build and evaluate Decision Tree model
def decision_tree_model(max_depth):
    # Instantiate a Decision Tree Classifier with specified max_depth
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    
    # Train the Decision Tree model on the training data
    dt.fit(X_train, y_train)

    # Predict using the trained model on the test data
    y_pred = dt.predict(X_test)

    # Calculate accuracy and generate a classification report
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)

    # Display information about the model and its performance
    print(f'Max Depth of Tree: {max_depth}')
    print(f'Decision Tree Accuracy: {accuracy}')
    print(f'Classification Report:\n{class_report}')
    print()

# Function to perform k-fold cross-validation on a given model
def k_fold_cross_validation(model, X, y, k):
    # Create k-fold cross-validation iterator
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    
    # Perform k-fold cross-validation on the model and data
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    # Display results of k-fold cross-validation
    print(f'K-Fold Cross-Validation with {k} folds:')
    print(f'Using Decision Tree Model with max depth {model.max_depth}')
    print(f'Accuracy Scores: {scores}')
    print(f'Mean Accuracy: {scores.mean()}')
    print()

# Run the Decision Tree model with different max_depth values
decision_tree_model(3)  # Evaluate the Decision Tree with max_depth = 3
decision_tree_model(5)  # Evaluate the Decision Tree with max_depth = 5
decision_tree_model(10)  # Evaluate the Decision Tree with max_depth = 10
decision_tree_model(15)  # Evaluate the Decision Tree with max_depth = 15 (additional)
decision_tree_model(20)  # Evaluate the Decision Tree with max_depth = 20 (additional)
decision_tree_model(25)  # Evaluate the Decision Tree with max_depth = 25 (additional)

# Example using k-fold cross-validation with Decision Tree
dt_5 = DecisionTreeClassifier(max_depth=5, random_state=42)  # Instantiate Decision Tree with max_depth = 5
dt_10 = DecisionTreeClassifier(max_depth=10, random_state=42)  # Instantiate Decision Tree with max_depth = 10
dt_15 = DecisionTreeClassifier(max_depth=15, random_state=42)  # Instantiate Decision Tree with max_depth = 15 (additional)
dt_20 = DecisionTreeClassifier(max_depth=20, random_state=42)  # Instantiate Decision Tree with max_depth = 20 (additional)

k_fold_cross_validation(dt_5, X, y, k=5)  # Perform k-fold cross-validation (k=5) using the Decision Tree model with max_depth = 5
k_fold_cross_validation(dt_10, X, y, k=5)  # Perform k-fold cross-validation (k=5) using the Decision Tree model with max_depth = 10
k_fold_cross_validation(dt_15, X, y, k=5)  # Perform k-fold cross-validation (k=5) using the Decision Tree model with max_depth = 15 (additional)
k_fold_cross_validation(dt_20, X, y, k=5)  # Perform k-fold cross-validation (k=5) using the Decision Tree model with max_depth = 20 (additional)

k_fold_cross_validation(dt_5, X, y, k=10)  # Perform k-fold cross-validation (k=10) using the Decision Tree model with max_depth = 5
k_fold_cross_validation(dt_10, X, y, k=10)  # Perform k-fold cross-validation (k=10) using the Decision Tree model with max_depth = 10
k_fold_cross_validation(dt_15, X, y, k=10)  # Perform k-fold cross-validation (k=10) using the Decision Tree model with max_depth = 15 (additional)
k_fold_cross_validation(dt_20, X, y, k=10)  # Perform k-fold cross-validation (k=10) using the Decision Tree model with max_depth = 20 (additional)