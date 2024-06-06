import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_profiling import ProfileReport

# Load the dataset
df = pd.read_csv('African_crises_dataset.csv')

# Display general information about the dataset
print(df.info())
print(df.describe())
print(df.head())

# Create a pandas profiling report
profile = ProfileReport(df, title='Systemic Crisis, Banking Crisis, Inflation Crisis In Africa by OG')
profile.to_file("African_crises_dataset_OG.html")

# Check for missing values
print(df.isnull().sum())

# Handle missing values (e.g., impute with mean or median)
df.fillna(df.mean(), inplace=True)

# Check for corrupted values (e.g., outliers, invalid data)
# Handle corrupted values (e.g., remove or transform)

# Check for duplicates
print(df.duplicated().sum())

# Remove duplicates
df.drop_duplicates(inplace=True)

# Check for outliers (e.g., using boxplots or scatter plots)
plt.boxplot(df['inflation_annual'])
plt.show()

# Handle outliers (e.g., remove or transform)

# Identify categorical features
categorical_features = df.select_dtypes(include=['object']).columns

# Encode categorical features using one-hot encoding or label encoding
df = pd.get_dummies(df, columns=categorical_features)

# Select the target variable
target = 'ystemic_crisis'

# Select the features
features = df.drop(columns=[target])

from sklearn.model_selection import train_test_split

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, df[target], test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier

# Select a machine learning classification algorithm (e.g., random forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training set
model.fit(X_train, y_train)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Make predictions on the test set
y_pred = model.predict(X_test)

# Assess the model performance using relevant evaluation metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Discuss alternative ways to improve model performance
# Some possible ways to improve model performance include:
# 1. Feature engineering: Creating new features or transforming existing ones to better capture the underlying patterns.
# 2. Hyperparameter tuning: Optimizing the parameters of the model to improve its performance.
# 3. Trying different algorithms: Experimenting with various classification algorithms to see which one performs better.
# 4. Ensemble methods: Using ensemble methods like bagging, boosting, or stacking to combine multiple models for better performance.
# 5. Addressing class imbalance: If the classes are imbalanced, techniques like oversampling, undersampling, or using algorithms that handle class imbalance well can be beneficial.
# 6. Cross-validation: Using cross-validation techniques to get a better estimate of the model's performance on unseen data.
# 7. Regularization: Applying regularization techniques to prevent overfitting and improve generalization.
# 8. Feature selection: Selecting the most relevant features to reduce noise and improve model performance.
# 9. Handling missing values and outliers more effectively.