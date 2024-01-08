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
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, silhouette_score
from sklearn.tree import DecisionTreeClassifier
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.svm import SVC
from sklearn.cluster import KMeans

def plt_property(title, xlabel, ylabel):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

df = pd.read_csv("/Users/apple/Desktop/uni/year3/IntroToAI/cw/star_classification.csv", na_values=['NA', '?'])

row_count = len(df.axes[0])
cols_count = len(df.axes[1])
print(f'The DataFrame has {row_count} rows.')
print(f'The DataFrame has {cols_count} columns.')

df = df.rename(columns={'g': 'Green Light', 'u': 'Ultraviolet Light', 'r': 'Red Light', 'i': 'Infrared Light', 'delta': 'Declination Angle', 'alpha': 'Ascension Angle'})

print(df.isnull().any())
df = df.dropna()
print(df.isnull().any())

print(df.tail())
print(df.head())

x = df[['redshift']]
y = df['class']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

smote = SMOTE(random_state=42)
print('Original data %s' % Counter(y))
x, y = smote.fit_resample(x, y)
print('Resampled data %s' % Counter(y))

redshift_data = df[['redshift']]
scaler = StandardScaler()
scaled_redshift = scaler.fit_transform(redshift_data)
df['scaled_redshift'] = scaled_redshift
print(pd.DataFrame({'Original': df['redshift'], 'Scaled': df['scaled_redshift']}).head())

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='class', y='scaled_redshift', palette='viridis')
plt_property('Scatter Plot of Redshift vs Class', 'Class', 'Scaled_Redshift')
plt.show()

X = df[['scaled_redshift']]
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def knn_model(nneighbours):
    knn = KNeighborsClassifier(n_neighbors=nneighbours)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred)
    print('Number of neighbours:', nneighbours)
    print('Baseline Model Accuracy:', accuracy)
    print('Classification Report:\n', class_report)
    print()

knn_model(3)
knn_model(5)
knn_model(10)
knn_model(100)
knn_model(math.isqrt(traindata_size))
knn_model(1000)

X = df.drop(['class', 'obj_ID', 'run_ID', 'rerun_ID','MJD', 'redshift'], axis=1)
df['class'] = label_encoder.fit_transform(df['class'])
y = keras.utils.to_categorical(df['class'])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def nn_model(n_hidden_layers, hidden_activation, n_hidden_units, n_epochs):
    print(f'Number of hidden layers: {n_hidden_layers}, Hidden layers activation type: {hidden_activation}')
    print(f'Number of hidden units: {n_hidden_units}, Number of epochs: {n_epochs}')
    model = Sequential()
    model.add(Dense(n_hidden_units, activation=hidden_activation, input_shape=(X_train.shape[1],)))
    for _ in range(1, n_hidden_layers):
        model.add(Dense(n_hidden_units, activation=hidden_activation))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    model.fit(X_train, y_train, verbose=2, epochs=n_epochs, batch_size=32)
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test accuracy: {accuracy}')
    print()

nn_model(1,'relu',64,10)
nn_model(1,'relu',64,200)
nn_model(1,'relu',1024,200)
nn_model(2,'relu',1024,200)

sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
X_test = sc.transform(X_test)
nn_model(2, 'relu', 64, 200)

def decision_tree_model(max_depth):
    dt = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, zero_division=0)
    print(f'Max Depth of Tree: {max_depth}')
    print(f'Decision Tree Accuracy: {accuracy}')
    print(f'Classification Report:\n{class_report}')
    print()

decision_tree_model(3)
decision_tree_model(5)
decision_tree_model(10)
decision_tree_model(15)
decision_tree_model(20)
decision_tree_model(25)

def k_fold_cross_validation(model, X, y, k):
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f'K-Fold Cross-Validation with {k} folds:')
    print(f'Using Decision Tree Model with max depth {model.max_depth}')
    print(f'Accuracy Scores: {scores}')
    print(f'Mean Accuracy: {scores.mean()}')
    print()

dt_5 = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_10 = DecisionTreeClassifier(max_depth=10, random_state=42)
dt_15 = DecisionTreeClassifier(max_depth=15, random_state=42)
dt_20 = DecisionTreeClassifier(max_depth=20, random_state=42)

k_fold_cross_validation(dt_5, X, y, 5)
k_fold_cross_validation(dt_10, X, y, 5)
k_fold_cross_validation(dt_15, X, y, 5)
k_fold_cross_validation(dt_20, X, y, 5)

k_fold_cross_validation(dt_5, X, y, 10)
k_fold_cross_validation(dt_10, X, y, 10)
k_fold_cross_validation(dt_15, X, y, 10)
k_fold_cross_validation(dt_20, X, y, 10)

def svm_model(kernel='linear', C=1.0):
    svm = SVC(kernel=kernel, C=C, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'SVM Model (Kernel: {kernel}, C: {C})')
    print(f'SVM Accuracy: {accuracy}')

    # Calculate and print the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print('Confusion Matrix:')
    print(cm)

    # Optional: Print classification report
    class_report = classification_report(y_test, y_pred, zero_division=0)
    print(f'Classification Report:\n{class_report}')
    print()

svm_model(kernel='linear', C=1.0)
svm_model(kernel='rbf', C=1.0)
svm_model(kernel='poly', C=1.0)

# K-Means Clustering
def k_means_model(n_clusters, data):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(data)
    print(f'Cluster Centers for {n_clusters} clusters:')
    print(kmeans.cluster_centers_)

    # Calculate Silhouette Score
    silhouette_avg = silhouette_score(data, labels)
    print(f'Silhouette Score for {n_clusters} clusters: {silhouette_avg}')

    return labels

cluster_data = df[['scaled_redshift', 'Green Light', 'Ultraviolet Light', 'Red Light', 'Infrared Light']]
labels = k_means_model(3, cluster_data)

# Visualizing Clusters (optional)
plt.figure(figsize=(10, 6))
plt.scatter(cluster_data.iloc[:, 0], cluster_data.iloc[:, 1], c=labels, cmap='viridis')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.show()
