import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, confusion_matrix, mean_squared_error, r2_score

# Load data
data = pd.read_csv('winequolity-white.csv')

# Data processing
seed = random.randint(10000,99999)
data.replace([-1000, 1000], np.nan, inplace=True)
data.fillna(data.median(), inplace=True)

# Removing outliers using the IQR method
def remove_outliers(df, columns):
    Q1 = df[columns].quantile(0.25)
    Q3 = df[columns].quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df[columns] < (Q1 - 1.5 * IQR)) | (df[columns] > (Q3 + 1.5 * IQR))).any(axis=1)]
    return df

data = remove_outliers(data, data.columns[:-1])  # Removing outliers for all columns except "quality"

# Displaying basic statistics to check for missing values
data_description = data.describe().isnull().sum()
print("Data description and checking for missing values:")
print(data_description)

# Creating histograms for each feature
data.hist(bins=30, figsize=(15, 10))
plt.tight_layout()
plt.show()

# Creating distribution plots for each feature
fig, axs = plt.subplots(4, 3, figsize=(15, 15))
axs = axs.flatten()
for i, col in enumerate(data.columns):
    sns.histplot(data[col], kde=True, ax=axs[i])
    axs[i].set_title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Creating boxplots for each feature
fig, axs = plt.subplots(4, 3, figsize=(15, 15))
axs = axs.flatten()
for i, col in enumerate(data.columns):
    sns.boxplot(y=data[col], ax=axs[i])
    axs[i].set_title(f'Boxplot of {col}')
plt.tight_layout()
plt.show()

# Correlation matrix
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pairplot for all features
sns.pairplot(data, diag_kind='kde')
plt.suptitle('Pairplot of All Features', y=1.02)
plt.show()

# Splitting into training and test sets
X = data.drop('quality', axis=1)
y = data['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Classification model
clf = RandomForestClassifier(random_state=seed)
clf.fit(X_train_scaled, y_train)
y_pred_clf = clf.predict(X_test_scaled)

# Classification report and confusion matrix
classification_rep = classification_report(y_test, y_pred_clf, zero_division=0)
conf_matrix = confusion_matrix(y_test, y_pred_clf)

print("Classification Report:")
print(classification_rep)
print("Confusion Matrix:")
print(conf_matrix)

# Regression model
reg = RandomForestRegressor(random_state=seed)
reg.fit(X_train_scaled, y_train)
y_pred_reg = reg.predict(X_test_scaled)
y_pred_reg_rounded = np.round(y_pred_reg)

# Regression metrics
mse = mean_squared_error(y_test, y_pred_reg)
r2 = r2_score(y_test, y_pred_reg)
print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Plotting actual vs predicted quality for the regression model
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_reg_rounded, alpha=0.3)
plt.xlabel('Actual Quality')
plt.ylabel('Predicted Quality')
plt.title('Actual vs Predicted Quality (Regression)')
plt.show()
