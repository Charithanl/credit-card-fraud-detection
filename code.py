# -- Imports --
import numpy as np #for numerical computations (e.g., array operations).
import pandas as pd #for data handling, loading CSV, and data manipulation.
import matplotlib.pyplot as plt #for plotting graphs.
import seaborn as sns #for more advanced, pretty statistical plots.

from sklearn.model_selection import train_test_split #splits dataset into training and testing sets.
from sklearn.preprocessing import StandardScaler #standardizes numerical features (mean=0, std=1).
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix # for model evaluation metrics

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# -- 1. Load dataset --
df = pd.read_csv('creditcard.csv')
print(df.shape) #shows number of rows and columns.
print(df.head()) #displays the first 5 rows.
print(df['Class'].value_counts()) #shows the count of fraudulent (1) and non-fraudulent (0) transactions — helps see class imbalance.
print(df['Amount'].describe()) #gives summary statistics for the transaction amounts (min, max, mean, std).

# -- 2. Data Exploration --
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Non-fraud Transactions')
plt.show()# Basic count plot
sns.countplot(x='Class', data=df)
plt.title('Fraud vs Non-fraud Transactions')
plt.show()

# --- Improved Plot 1: Log scale with count + percentage annotations ---
class_counts = df['Class'].value_counts()
total = len(df)

plt.figure(figsize=(6,4))
ax = sns.countplot(x='Class', data=df, palette='viridis')
plt.yscale('log')
plt.title('Fraud vs Non-fraud Transactions (Log Scale)', fontsize=12, fontweight='bold')
plt.xlabel('Class (0 = Non-fraud, 1 = Fraud)', fontsize=10)
plt.ylabel('Count (log scale)', fontsize=10)

# Annotate counts and percentages
for p in ax.patches:
    count = int(p.get_height())
    percentage = 100 * count / total
    ax.annotate(f'{count}\n({percentage:.3f}%)',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=9, color='black')

plt.tight_layout()
plt.show()

#histogram of transaction amounts.
sns.histplot(df['Amount'], bins=50, kde=True)
plt.title('Transaction Amount distribution')
plt.show()

# -- 4. Data Preprocessing / Manipulation --
scaler = StandardScaler()
df['Scaled_Amount'] = scaler.fit_transform(df[['Amount']])
df2 = df.drop(['Time','Amount'], axis=1)
X = df2.drop('Class', axis=1)
y = df2['Class']

# -- 5. Train-Test Split --
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123, stratify=y) #Splits data into training (80%) and testing (20%) sets.
print(X_train.shape, X_test.shape)

# -- 6. Logistic Regresssion Model --
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
y_proba_lr =lr.predict_proba(X_test)[:,1]
auc_lr = roc_auc_score(y_test, y_proba_lr)
print("Logistic Regression AUC:", auc_lr)
print(classification_report(y_test, y_pred_lr))
fpr, tpr, _ = roc_curve(y_test, y_proba_lr)
plt.plot(fpr, tpr, label=f'LR (AUC = {auc_lr:.3f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
