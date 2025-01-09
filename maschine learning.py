# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, silhouette_score
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("student_lifestyle_dataset.csv")

# Display the first few rows of the dataset
print("Dataset Head:")
print(df.head())

# Display basic information about the dataset
print("\nDataset Info:")
print(df.info())

# Display descriptive statistics
print("\nDescriptive Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Convert categorical 'Stress_Level' to numerical
df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Moderate': 1, 'High': 2})

# Display the updated dataset
print("\nUpdated Dataset Head:")
print(df.head())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Features and target variable for regression
X_reg = df[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'Stress_Level']]
y_reg = df['GPA']

# Features and target variable for classification
X_clf = df[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day']]
y_clf = df['Stress_Level']

# Features for clustering
X_clust = df[['Study_Hours_Per_Day', 'Sleep_Hours_Per_Day', 'Physical_Activity_Hours_Per_Day', 'GPA']]

# Standardize features for clustering
scaler = StandardScaler()
X_clust_scaled = scaler.fit_transform(X_clust)

# Split the data into training and testing sets for regression and classification
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

# Regression: Linear Regression
reg_model = LinearRegression()
reg_model.fit(X_train_reg, y_train_reg)
y_pred_reg = reg_model.predict(X_test_reg)

# Evaluate regression model
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print("\nRegression Model Evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R2 Score: {r2:.4f} ({(r2 * 100):.2f}%)")

# Classification: Logistic Regression
clf_model = LogisticRegression(max_iter=1000)
clf_model.fit(X_train_clf, y_train_clf)
y_pred_clf = clf_model.predict(X_test_clf)

# Evaluate classification model
accuracy = accuracy_score(y_test_clf, y_pred_clf)
conf_matrix = confusion_matrix(y_test_clf, y_pred_clf)
print("\nClassification Model Evaluation:")
print(f"Accuracy: {accuracy:.4f} ({(accuracy * 100):.2f}%)")
print("Confusion Matrix:")
print(conf_matrix)

# Clustering: KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_clust_scaled)

# Evaluate clustering model
silhouette_avg = silhouette_score(X_clust_scaled, df['Cluster'])
print("\nClustering Model Evaluation:")
print(f"Silhouette Score: {silhouette_avg:.4f}")

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Study_Hours_Per_Day', y='GPA', hue='Cluster', data=df, palette='viridis')
plt.title("Clustering: Study Hours vs GPA")
plt.xlabel("Study Hours Per Day")
plt.ylabel("GPA")
plt.show()

# Scatter plot: Actual vs Predicted GPA (Regression)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_reg, y=y_pred_reg, color='blue')
plt.plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], color='red', linestyle='--')
plt.title("Actual vs Predicted GPA (Regression)")
plt.xlabel("Actual GPA")
plt.ylabel("Predicted GPA")
plt.show()

# Confusion Matrix Heatmap (Classification)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Low', 'Moderate', 'High'], yticklabels=['Low', 'Moderate', 'High'])
plt.title("Confusion Matrix (Classification)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()