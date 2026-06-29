# 🫘 Dry Bean Classification

A machine learning pipeline that classifies dry bean varieties from morphological image features.  
The project covers end-to-end steps: exploratory data analysis, preprocessing, multi-model training, and evaluation.

---

## 📁 Project Structure

```
dry-bean-classification/
│
├── Dry_Bean_Dataset.csv       # Raw dataset
├── EDA.ipynb                  # Exploratory Data Analysis
├── modeling.ipynb             # Preprocessing + model training + evaluation
│
├── XGBoost_model.joblib  # Saved best model
├── power_transformer.joblib   # Fitted Yeo-Johnson transformer
├── pca.joblib                 # Fitted PCA (10 components)
└── label_encoder.joblib       # Fitted LabelEncoder
```

---

## 📊 Dataset

| Property        | Value                                              |
|-----------------|----------------------------------------------------|
| Source          | [UCI / Kaggle — Dry Bean Dataset](https://www.kaggle.com/datasets/muratkokludataset/dry-bean-dataset) |
| Instances       | 13,611                                             |
| Features        | 16 numerical (area, perimeter, compactness, shape factors, …) |
| Target classes  | 7 (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira) |
| Missing values  | None                                               |

---

## 🔍 Exploratory Data Analysis (`EDA.ipynb`)

| Topic                  | Finding                                                       |
|------------------------|---------------------------------------------------------------|
| Class balance          | Moderate imbalance across 7 classes                          |
| Skewness               | Several features exhibit high right-skew                     |
| Outliers               | Scattered across classes — handled per class via IQR         |
| Multicollinearity      | Multiple feature pairs with \|r\| > 0.90                    |
| Top discriminators     | Identified via Random Forest feature importance              |
| Hard class pair        | SIRA & DERMASON overlap most in feature space                |

---

## ⚙️ Preprocessing Pipeline (`modeling.ipynb`)

All transformers are **fit exclusively on the training set** to prevent data leakage.

```
Raw data
   │
   ├─ train_test_split (80 / 20, stratified)
   │
   ├─ Outlier handling   — IQR per class; replace with per-class median
   │
   ├─ Yeo-Johnson        — PowerTransformer (standardize=True)
   │                        corrects skewness + scales in one step
   │
   ├─ PCA                — 10 components (≥ 95 % explained variance)
   │                        reduces multicollinearity, speeds up SVM & XGBoost
   │
   └─ SMOTE              — applied after PCA on train only
                           balances class distribution for training
```

> **Why SMOTE after PCA?**  
> Synthesising samples in the original 16-D space and then projecting can distort the synthetic points.  
> Applying SMOTE in PCA space keeps the geometry consistent.

---

## 🤖 Models & Hyperparameter Tuning

Hyperparameters were selected via `RandomizedSearchCV` (50 iterations, 5-fold CV).  
The search code is preserved (commented out) in the notebook for reproducibility.

### XGBoost
| Parameter         | Value |
|-------------------|-------|
| `n_estimators`    | 100   |
| `max_depth`       | 7     |
| `learning_rate`   | 0.3   |
| `subsample`       | 0.8   |
| `colsample_bytree`| 0.8   |
| `gamma`           | 0.1   |
| Best CV accuracy  | 0.9756 |

### SVM (RBF kernel)
| Parameter      | Value     |
|----------------|-----------|
| `C`            | 100       |
| `kernel`       | rbf       |
| `gamma`        | 0.1       |
| `class_weight` | balanced  |
| Best CV accuracy | 0.9740  |

### Random Forest
| Parameter           | Value    |
|---------------------|----------|
| `n_estimators`      | 300      |
| `max_depth`         | 20       |
| `max_features`      | sqrt     |
| `class_weight`      | balanced |
| Best CV accuracy    | 0.9757   |

---

## 📈 Results

### Test Set Performance

| Model             | Accuracy | Precision | Recall | F1 Score | AUC    |
|-------------------|:--------:|:---------:|:------:|:--------:|:------:|
| **XGBoost**       | **0.9479** | **0.9481** | **0.9479** | **0.9479** | 0.9962 |
| SVM               | 0.9475   | 0.9477    | 0.9475 | 0.9475   | **0.9965** |
| Random Forest     | 0.9479   | 0.9480    | 0.9479 | 0.9479   | 0.9962 |

### Cross-Validation (5-fold, stratified)

| Model             | Mean Accuracy | Std    |
|-------------------|:-------------:|:------:|
| XGBoost           | 0.9306        | 0.0106 |
| **SVM**           | **0.9365**    | 0.0164 |
| Random Forest     | 0.9298        | 0.0113 |

### 🏆 Best Model: XGBoost

XGBoost ties with Random Forest on test-set Accuracy and F1 (0.9479), and is selected as the best model by F1 Score. All three models perform within 0.05% of each other on the test set, indicating a robust and well-generalising pipeline. SVM leads on cross-validation mean accuracy (0.9365) but shows slightly higher variance (std 0.0164).

---

## 🔮 Inference

To predict on new data, apply transformations in this exact order:

```python
import joblib
import pandas as pd

# Load saved objects
pt    = joblib.load("power_transformer.joblib")
pca   = joblib.load("pca.joblib")
le    = joblib.load("label_encoder.joblib")
model = joblib.load("XGBoost_model.joblib")

# Transform new samples
X_new_trans = pt.transform(X_new)          # 1. Yeo-Johnson
X_new_pca   = pca.transform(X_new_trans)   # 2. PCA
y_pred_enc  = model.predict(X_new_pca)     # 3. Predict
y_pred      = le.inverse_transform(y_pred_enc)  # 4. Decode labels

print(y_pred)
```

---

## 🛠️ Requirements

```
numpy
pandas
matplotlib
seaborn
scikit-learn
imbalanced-learn
xgboost
joblib
```

Install all dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost joblib
```

---

## 🚀 Getting Started

```bash
# Clone the repository
git clone https://github.com/Ali-sarafraz/dry-bean-classification.git
cd dry-bean-classification

# Install dependencies
pip install -r requirements.txt

# Run EDA
jupyter notebook EDA.ipynb

# Run the modelling pipeline
jupyter notebook modeling.ipynb
```
