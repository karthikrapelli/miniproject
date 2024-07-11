import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load and preprocess data
df = pd.read_csv('bmi.csv')

encoding = {
    'Obese Class 1': 3,
    'Overweight': 2,
    'Underweight': 0,
    'Obese Class 2': 4,
    'Obese Class 3': 5,
    'Normal Weight': 1
}

df['BmiDesc_encoded'] = df['BmiClass'].map(encoding)

# Prepare features and target arrays
X = df[['Height', 'Weight']].values
print("Number of features:", X.shape[1])  # Print the number of features
y = df['BmiDesc_encoded'].values

# Split the data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning to find the best value of n
n_values = list(range(1, 30))
accuracies = []

for n in n_values:
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    y_pred_val = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred_val)
    accuracies.append(accuracy)

best_n = n_values[np.argmax(accuracies)]

# Train the best KNN classifier on the entire training set
knn_best = KNeighborsClassifier(n_neighbors=best_n)
knn_best.fit(X_train, y_train)

def predict(input_data):
    return knn_best.predict(input_data)
