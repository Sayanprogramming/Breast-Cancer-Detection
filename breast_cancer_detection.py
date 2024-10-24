import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.datasets
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# loading the data from sklearn
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# loading the data into a dataframe
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

# adding the 'target' column to the data frame
data_frame['label'] = breast_cancer_dataset.target

# print the first 5 rows of the dataframe
print(data_frame.head())

# print last 5 rows of the dataframe
print(data_frame.tail())

# print number of rows and columns in the dataset
print(data_frame.shape)

# getting some information about the data
data_frame.info()

# checking for missing values
print(data_frame.isnull().sum())

# statistical measures about the data
print(data_frame.describe())

# checking the distribution of Target Varibale
print(data_frame['label'].value_counts())

# group by label to find the mean values per class
print(data_frame.groupby('label').mean())

# Separating the features and target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']

# print X and Y
print(X)
print(Y)

# Splitting the data into training data & Testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# print the shapes of the train and test sets
print(X.shape, X_train.shape, X_test.shape)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initializing and training the model
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, Y_train)

# Print the best parameters (this will print the default parameters used in the model)
print("Model Parameters:", rf_model.get_params())

# Prediction using the model
y_pred = rf_model.predict(X_test)

#check the Cross-validated accuracy of the model
cv_scores = cross_val_score(rf_model, X, Y, cv=5, scoring='accuracy')
print(f"Cross-validated Accuracy: {cv_scores.mean():.2f}")

# Evaluate the model
print("Accuracy:", accuracy_score(Y_test, y_pred))
print(classification_report(Y_test, y_pred))
print("Accuracy of the Model is: ", accuracy_score(Y_test, y_pred))

#Confusion Matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

# Flatten the confusion matrix
cm_flat = conf_matrix.flatten()

# Create labels for each confusion matrix element
labels = ['True Negatives', 'False Positives', 'False Negatives', 'True Positives']

# Plot bar chart and visualize the confusion matrix
plt.bar(labels, cm_flat, color=['blue', 'orange', 'red', 'green'])
plt.title('Confusion Matrix')
plt.ylabel('True Count')
plt.xlabel('Predicted Categories')
plt.show()
importances = rf_model.feature_importances_
feature_names = X.columns

# importance of different features in a dataset using a bar plot
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10,6))
plt.title("Feature Importance of Each Dataset")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.show()

# Example of Input Data Using for predict 
input_data = (17.99, 10.38, 122.80, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 1.095, 0.9053, 8.589, 153.40, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 184.60, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.11890)

# Create a DataFrame for the input data with the correct feature names
input_data_df = pd.DataFrame([input_data], columns=X.columns)

# Standardize the input data using the same scaler
input_data_std = scaler.transform(input_data_df)

# Make prediction
prediction = rf_model.predict(input_data_std)

# Output the prediction result
result = 'Malignant Tumor' if prediction[0] == 0 else 'Primary Tumor Location'
print(f'The Tumor is {result}')

