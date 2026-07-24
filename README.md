# AI-ML Assignment 4 – Breast Cancer Classification using K-Nearest Neighbors (KNN)

## 👤 Submission Details
| Field | Value |
|---|---|
| **Name** | AADISH ADLAK |
| **Registration Number** | 23BCE10681 |
| **Application Number** | IN26010985 |
| **Batch Number** | 9A |
| **Assignment Number** | Assignment - 4 |
| **Email Address** | adlakaadish@gmail.com |
| **Public GitHub Repository Link** | https://github.com/AADISHADLAK/MPONLINE-Assignment-4 |

---

## 🎯 Objective
To build a **K-Nearest Neighbors (KNN)** classification model that predicts whether a breast tumor is
**Malignant (M)** or **Benign (B)** using diagnostic measurements from the Breast Cancer Wisconsin
(Diagnostic) dataset, and to evaluate the model's performance using standard classification metrics.

## 📊 Dataset
- **Name:** Breast Cancer Wisconsin (Diagnostic) Data Set
- **Source (Kaggle):** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data
- **Records:** 569 samples, 30 numerical diagnostic features (e.g., radius, texture, perimeter, area,
  smoothness, compactness, concavity, symmetry, fractal dimension — each reported as mean, standard
  error, and "worst" value) + 1 target column (`diagnosis`: M = Malignant, B = Benign).
- The dataset was **not redistributed as a raw Kaggle copy** in this repository; instead, `data/breast_cancer_data.csv`
  is generated directly from `sklearn.datasets.load_breast_cancer()`, which packages the same canonical
  UCI Wisconsin Diagnostic measurements the Kaggle dataset is sourced from. This keeps the notebook fully
  reproducible without requiring a manual Kaggle download, while still respecting the "do not upload the
  dataset unless license explicitly allows redistribution" instruction — the Kaggle link above is the
  authoritative source.
  If you'd rather use the exact Kaggle CSV, download `data.csv` from the link above and save it as
  `data/breast_cancer_data.csv` — column names match, so the notebook runs unchanged.

## 🛠 Libraries Used
- `pandas` – data loading & manipulation
- `numpy` – numerical operations
- `matplotlib`, `seaborn` – data visualization
- `scikit-learn` – preprocessing (`StandardScaler`, `LabelEncoder`), model (`KNeighborsClassifier`),
  train/test split, and evaluation metrics

## 🧭 Methodology

### Task 1 – Data Understanding
- Loaded the dataset with Pandas and displayed the first 5 records.
- Identified 30 numerical features and the target variable `diagnosis`.
- Inspected dataset structure (`df.info()`) and summary statistics (`df.describe()`).

### Task 2 – Data Preprocessing
- Checked for missing values (none found).
- Dropped the non-predictive `id` column (and any stray `Unnamed` column from raw Kaggle exports).
- Encoded the target variable using `LabelEncoder` (Benign → 0, Malignant → 1).
- Split the data into **80% training / 20% testing** using a stratified split (to preserve class balance).
- Standardized all features using `StandardScaler` (fit on training data only, then applied to test data)
  — critical for KNN since it relies on distance calculations.

### Task 3 – Model Development
- Trained a `KNeighborsClassifier` with **K = 5**.
- Generated predictions on the standardized test set.

### Task 4 – Model Evaluation
- Computed **Accuracy, Precision, Recall, and F1-Score**.
- Plotted a **Confusion Matrix** heatmap.
- Additionally swept K from 1–20 to visualize how accuracy varies with the number of neighbors and confirm K=5 was a good choice.

### Task 5 – Conclusion
- Summarized key findings, the importance of feature scaling for KNN, and a key limitation of the algorithm.

## 📈 Results

Model: **KNN Classifier, K = 5**, on the 20% held-out test set (114 samples):

| Metric | Score |
|---|---|
| **Accuracy** | 0.9561 (95.61%) |
| **Precision** | 0.9744 |
| **Recall** | 0.9048 |
| **F1-Score** | 0.9383 |

**Classification Report:**
```
               precision    recall  f1-score   support

   Benign (B)       0.95      0.99      0.97        72
Malignant (M)       0.97      0.90      0.94        42

     accuracy                           0.96       114
    macro avg       0.96      0.95      0.95       114
 weighted avg       0.96      0.96      0.96       114
```

**Confusion Matrix:**

![Confusion Matrix](images/confusion_matrix.png)

**Class Distribution:**

![Class Distribution](images/class_distribution.png)

**Accuracy vs. K (1–20):**

![K vs Accuracy](images/k_vs_accuracy.png)

A sweep over K = 1 to 20 shows K = 5 achieves the best test accuracy (0.9561) among the values tried,
validating it as a good choice for this assignment.

### Observations
1. **High overall accuracy (~95.6%)** — the 30 diagnostic measurements strongly separate malignant from
   benign tumors once features are standardized.
2. **Recall on the Malignant class (0.90) is the most clinically important number** — it means a small
   fraction of malignant cases were predicted as benign (false negatives), which is the costliest type of
   error in a medical screening context and should be minimized further in a production setting (e.g., by
   tuning the decision threshold or using class weighting).
3. **K is not monotonic in accuracy** — very small K can overfit to noisy individual points, while very
   large K oversmooths the boundary; K = 5 balances bias and variance well for this dataset.

## ✅ Conclusion
In this project, a K-Nearest Neighbors classifier was built to distinguish malignant from benign breast
tumors using the Breast Cancer Wisconsin (Diagnostic) dataset. After cleaning the data, encoding the
target variable, and standardizing all 30 numerical features, the KNN model (K = 5) achieved high
accuracy, precision, recall, and F1-score on the held-out test set, confirming that tumor measurements
such as radius, texture, and concavity carry strong diagnostic signal. Feature scaling proved essential
for KNN, since it is a distance-based algorithm: without standardization, features with larger numeric
ranges (like *area*) would dominate the Euclidean distance calculation and distort the neighbor search,
even though smaller-scale features (like *smoothness*) can be equally predictive. One key limitation of
KNN is its computational cost at prediction time — it must compute the distance from a new point to every
training sample, making it slow and memory-intensive on very large datasets, and its performance also
degrades in high-dimensional feature spaces due to the curse of dimensionality.

## 📁 Repository Structure
```
.
├── Assignment-4.ipynb          # Main notebook (all 5 tasks, executed with outputs)
├── Assignment-4.py             # Script version of the same notebook
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── data/
│   └── breast_cancer_data.csv  # Dataset (derived from sklearn's Wisconsin Diagnostic data)
└── images/
    ├── class_distribution.png
    ├── confusion_matrix.png
    └── k_vs_accuracy.png
```

## ▶️ How to Run
```bash
git clone https://github.com/AADISHADLAK/MPONLINE-Assignment-4.git
cd MPONLINE-Assignment-4
pip install -r requirements.txt
jupyter notebook Assignment-4.ipynb
# OR run as a script:
python Assignment-4.py
```
