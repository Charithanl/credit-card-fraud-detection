# 💳 Credit Card Fraud Detection using Machine Learning

### 📘 Project Overview

This project focuses on **detecting fraudulent credit card transactions** using multiple machine learning algorithms.
It leverages the popular **Credit Card Fraud Detection Dataset** (European transactions, September 2013) and compares the performance of various models on a highly **imbalanced classification problem**.

---

## 📂 Dataset Summary

* **Total Transactions:** 284,807
* **Features:** 31 (including anonymized PCA features `V1–V28`)
* **Target Variable:** `Class` → 0 = Non-fraud, 1 = Fraud
* **Fraud Cases:** 492 (≈0.17%)
* **Non-fraud Cases:** 284,315

📊 The dataset is **highly imbalanced**, meaning accuracy alone is not a reliable performance metric — hence, metrics like **AUC, Recall, and F1-score** are emphasized.

---

## 🧾 **Section 1 – Load Dataset**

**Code Summary:**

```python
df = pd.read_csv('creditcard.csv')
print(df.shape)
print(df.head())
print(df['Class'].value_counts())
print(df['Amount'].describe())
```

**Output Overview:**

* Shape: `(284807, 31)`
* Legitimate transactions: `284,315`
* Fraudulent transactions: `492`
* Average amount: `€88.35` (max ≈ €25,691)

**Key Insights:**

* Severe **class imbalance (~0.17% fraud)**.
* Transaction amounts are **right-skewed** (most transactions are small).
* PCA-transformed features (`V1–V28`) ensure privacy.

---

## 🔍 **Section 2 – Data Exploration**

### **Fraud vs Non-Fraud Visualization**

![Fraud vs Non-Fraud (Log Scale)](C:\credit-card-fraud-detection\images\sec1a.png)

(C:\credit-card-fraud-detection\images\section 1b.png)

**Interpretation:**

* 99.827% of transactions are **legitimate**, only 0.173% are **fraudulent**.
* Logarithmic scale helps visualize the imbalance effectively.

---

### **Transaction Amount Distribution**

![Transaction Amount Distribution]

C:\credit-card-fraud-detection\images\histogram plot.png

**Insights:**

* Most transactions are **small (< €100)**.
* A few large transactions create a **long tail**.
* This feature requires **scaling** for modeling.

---

## ⚙️ **Section 4 – Data Preprocessing / Manipulation**

**Steps:**

* Standardized `Amount` using `StandardScaler()`.
* Dropped unnecessary columns: `Time`, `Amount`.
* Created:

  * `X` → feature set (29 columns)
  * `y` → target (`Class`)

✅ Result: Dataset cleaned, normalized, and ready for training.

---

## ✂️ **Section 5 – Train-Test Split**

**Code:**

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=123, stratify=y)
```

**Output:**

```
X_train: (227,845, 29)
X_test: (56,962, 29)
```

**Notes:**

* 80% training / 20% testing split.
* `stratify=y` maintains the same class ratio.
* Ensures model evaluation consistency.

---

## 🧮 **Section 6 – Logistic Regression Model**

![ROC – Logistic Regression]

C:\credit-card-fraud-detection\images\Lr PLOT.png

**Metrics:**

| Metric                | Value |
| --------------------- | ----- |
| **AUC**               | 0.961 |
| **Precision (Fraud)** | 0.86  |
| **Recall (Fraud)**    | 0.57  |
| **F1-score (Fraud)**  | 0.69  |

**Insights:**

* Excellent **AUC (0.96)** — strong ranking performance.
* Moderate recall (misses some frauds).
* Strong precision (few false positives).
   **Solid baseline model.**

---

##  **Section 7 – Decision Tree Classifier**

![ROC – Decision Tree]

C:\credit-card-fraud-detection\images\DT plot.png

**Metrics:**

| Metric                | Value |
| --------------------- | ----- |
| **AUC**               | 0.857 |
| **Precision (Fraud)** | 0.71  |
| **Recall (Fraud)**    | 0.71  |
| **F1-score (Fraud)**  | 0.71  |

**Insights:**

* Recall improved from 0.57 → 0.71.
* Slightly lower AUC — possible overfitting.
* More balanced fraud detection performance.

---

## 🤖 **Section 8 – Artificial Neural Network (MLP Classifier)**

![ROC – MLP Classifier]

C:\credit-card-fraud-detection\images\MLP plot.png

**Metrics:**

| Metric                | Value |
| --------------------- | ----- |
| **AUC**               | 0.936 |
| **Precision (Fraud)** | 0.86  |
| **Recall (Fraud)**    | 0.72  |
| **F1-score (Fraud)**  | 0.78  |

**Insights:**

* Best recall and F1 performance among all models.
* Slightly lower AUC than Logistic Regression but higher recall.
  ✅ **Best overall fraud detection model.**

---

## 🚀 **Section 9 – Gradient Boosting (HistGradientBoosting)**

C:\credit-card-fraud-detection\images\hgb plot.png

**Metrics:**

| Metric                | Value |
| --------------------- | ----- |
| **AUC**               | 0.832 |
| **Precision (Fraud)** | 0.42  |
| **Recall (Fraud)**    | 0.59  |
| **F1-score (Fraud)**  | 0.49  |

**Insights:**

* Moderate AUC (0.83) — decent start but needs tuning.
* Higher recall than Logistic Regression, lower precision.
* Benefits from **hyperparameter tuning** and **class balancing**.

---

## 📊 **Model Performance Comparison**

| Model                    | AUC   | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | Remarks                        |
| ------------------------ | ----- | ----------------- | -------------- | ---------- | ------------------------------ |
| **Logistic Regression**  | 0.961 | 0.86              | 0.57           | 0.69       | Strong baseline, excellent AUC |
| **Decision Tree**        | 0.857 | 0.71              | 0.71           | 0.71       | Balanced, interpretable        |
| **MLP Classifier**       | 0.936 | 0.86              | 0.72           | 0.78       | Best overall fraud detection   |
| **HistGradientBoosting** | 0.832 | 0.42              | 0.59           | 0.49       | Decent recall, needs tuning    |

---

##  **Key Takeaways**

* **MLP Classifier** achieved the best fraud detection performance.
* **Logistic Regression** remains a strong, interpretable baseline.
* **Ensemble methods** can improve with tuning and resampling.

### Future Improvements

* Apply **SMOTE** for synthetic oversampling.
* Use **cross-validation** for stable results.
* Explore **cost-sensitive learning** to penalize false negatives.

---

 **Tech Stack**

* **Language:** Python 🐍
* **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, scikit-learn
* **Models:** Logistic Regression, Decision Tree, MLPClassifier, HistGradientBoostingClassifier
* **Metrics:** ROC-AUC, Precision, Recall, F1-score, Confusion Matrix

---

**Conclusion**

Fraud detection is a **highly imbalanced classification problem** where accuracy alone can be misleading.
Through experimentation with multiple models, the **MLP Classifier** offered the **best trade-off** between recall and precision, effectively identifying fraudulent activities.

This project emphasizes:

* Data preprocessing and scaling
* Stratified sampling
* Using AUC, recall, and F1 as performance metrics
* Handling class imbalance stratergically

Author
Project by: Charitha NL
Dataset Source: Kaggle – Credit Card Fraud Detection (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
License: MIT