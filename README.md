Importing Required Libraries:


import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sklearn.datasets

from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

NumPy: Used for handling arrays and numerical operations.

Pandas: Used for data manipulation and analysis.

Matplotlib: Used for data visualization.

Scikit-learn: Provides tools for machine learning, including model selection, feature scaling, classification, and evaluation.


Loading the Breast Cancer Dataset:


breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

The load_breast_cancer() function from sklearn.datasets loads a built-in breast cancer dataset containing 30 numerical features and a target variable (label).


Creating DataFrame and Adding Target:


data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)

data_frame['label'] = breast_cancer_dataset.target

A DataFrame is created using the feature data, and the target variable (label) is added to it.


Exploring the Data:


Print the first and last 5 rows of the data to visually inspect:

print(data_frame.head())

print(data_frame.tail())


Print the shape of the data (number of rows and columns):


print(data_frame.shape)


Get an overview of the data types and any missing values:


data_frame.info()

print(data_frame.isnull().sum())


Summarize the data using descriptive statistics:


print(data_frame.describe())

Examine the distribution of the target variable (malignant vs initial):


print(data_frame['label'].value_counts())


Group the data by label to see the mean of each feature per class:


print(data_frame.groupby('label').mean())


Separating Features and Target:


X = data_frame.drop(columns='label', axis=1)

Y = data_frame['label']

The features (X) and target (Y) are separated. The target represents the diagnosis (0 = malignant, 1 = primary).


Splitting the Data into Training and Testing Sets:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

The dataset is split into training (80%) and testing (20%) sets using train_test_split() to ensure fair evaluation of the model.


Feature Scaling:


scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)

StandardScaler standardizes features by removing the mean and scaling them to unit variance. This ensures the model treats all features equally.


Training the Random Forest Classifier:


rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, Y_train)

A Random Forest classifier is initialized and trained using the training data.

Random forest is an ensemble method that creates multiple decision trees and averages their predictions for more accurate results.


Cross-Validation:


cv_scores = cross_val_score(rf_model, X, Y, cv=5, scoring='accuracy')

Cross-validation (5-fold) is used to evaluate the model's performance across different subsets of the data to prevent overfitting.


Model Prediction and Evaluation:


y_pred = rf_model.predict(X_test)

print("Accuracy:", accuracy_score(Y_test, y_pred))

print(classification_report(Y_test, y_pred))

The model predicts the outcomes for the test data (X_test), and accuracy, precision, recall, and F1-score are calculated.


Confusion Matrix:


conf_matrix = confusion_matrix(Y_test, y_pred)

The confusion matrix gives a clear view of true positive, true negative, false positive, and false negative predictions.


Visualizing the Confusion Matrix:


cm_flat = conf_matrix.flatten()

plt.bar(labels, cm_flat, color=['blue', 'orange', 'red', 'green'])

plt.title('Confusion Matrix')

plt.ylabel('True Count')

plt.xlabel('Predicted Categories')

plt.show()

A bar chart is used to visualize the confusion matrix.


Feature Importance:


importances = rf_model.feature_importances_

plt.bar(range(X.shape[1]), importances[indices], align="center")

plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)

plt.show()

Feature importance provides insight into which features contribute the most to the model's decision-making.


Predicting with New Data:


input_data = (17.99, 10.38, 122.80, 1001.0, ...)

input_data_df = pd.DataFrame([input_data], columns=X.columns)

input_data_std = scaler.transform(input_data_df)

prediction = rf_model.predict(input_data_std)


Output the prediction result:


result = 'Malignant Tumor' if prediction[0] == 0 else 'Primary Tumor Location'

print(f'The Tumor is {result}')
