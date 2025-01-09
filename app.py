# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Load dataset
df = pd.read_csv('student_lifestyle_dataset.csv')


# 1. Scatter Plot: Study Hours vs GPA by Stress Level
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Study_Hours_Per_Day', y='GPA', hue='Stress_Level', palette='Set1')
plt.title("Study Hours vs GPA by Stress Level")
plt.xlabel("Study Hours Per Day")
plt.ylabel("GPA")
plt.legend(title="Stress Level")
plt.show()  

# 2. Bar Chart: Stress Level Distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Stress_Level', palette='Set2')
plt.title("Stress Level Distribution")
plt.xlabel("Stress Level")
plt.ylabel("Number of Students")
plt.show()  


# 3. Boxplot: GPA Distribution by Stress Level
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Stress_Level', y='GPA', palette='Set2')
plt.title("GPA Distribution by Stress Level")
plt.xlabel("Stress Level")
plt.ylabel("GPA")
plt.show()  

# 4. Pie Chart: Logical Result Distribution
plt.figure(figsize=(6, 6))
logical_result_counts = df['Is_Logical_Result'].value_counts()
plt.pie(logical_result_counts, labels=logical_result_counts.index, autopct='%1.1f%%', colors=['lightgreen', 'lightcoral'])
plt.title("Logical Result Distribution")
plt.show()  

# 5. Pairplot: Relationships Between Study Hours, Extracurricular Hours, GPA, and Stress Level
sns.pairplot(df[['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'GPA', 'Stress_Level']], hue='Stress_Level')
plt.show()  

# Encoding categorical variables
df['Stress_Level'] = df['Stress_Level'].map({'Low': 0, 'Medium': 1, 'High': 2})  # Example mapping
df['Is_Logical_Result'] = df['Is_Logical_Result'].map({0: 'No', 1: 'Yes'})

# Splitting data into features and target variables
X = df[['Study_Hours_Per_Day', 'Extracurricular_Hours_Per_Day', 'Stress_Level']]  # Feature columns
y_reg = df['GPA']  # Target for regression
y_clf = df['Is_Logical_Result'].map({'Yes': 1, 'No': 0})  # Target for classification

# Split data into training and testing sets
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

