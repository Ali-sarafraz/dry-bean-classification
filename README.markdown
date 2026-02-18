# Dry Bean Classification Project

## Overview
This project focuses on classifying dry bean types based on features provided in the `dry_bean_dataset.csv` dataset. The project includes exploratory data analysis (EDA), model training, and evaluation using multiple machine learning algorithms, with XGBoost identified as the best-performing model.

## Project Structure
- `dry_bean_dataset.csv`: The dataset containing features and labels for dry bean classification.
- `EDA.ipynb`: Jupyter notebook for exploratory data analysis, including data visualization and preprocessing steps.
- `modeling.ipynb`: Jupyter notebook for training and evaluating machine learning models (SVM, Random Forest, and XGBoost).
- `requirements.txt`: List of Python dependencies required to run the project.
- `RandomForest_model.joblib`: Saved Random Forest model, which outperformed SVM and XGBoost in evaluation.

## Methodology
1. **Exploratory Data Analysis (EDA)**:
   - Conducted in `EDA.ipynb`.
   - Includes data cleaning, feature analysis, and visualizations to understand the dataset's characteristics.

2. **Modeling**:
   - Implemented in `modeling.ipynb`.
   - Three machine learning models were trained and evaluated:
     - **Support Vector Machine (SVM)**: Baseline classification model with the highest cross-validation mean accuracy, but slightly weaker test-set performance.
     - **Random Forest**: Ensemble model that delivered the best overall test-set performance (Accuracy, Precision, Recall, F1).
     - **XGBoost**: Gradient boosting model with strong, consistent results and the highest AUC, performing very close to Random Forest.

3. **Model Selection**:
   - Random forest was selected as the final model due to its superior performance.
   - The trained Random forest model is saved as `RandomForest_model.joblib` for reproducibility and deployment.

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have Jupyter Notebook or JupyterLab installed to run the notebooks:
   ```bash
   pip install jupyter
   ```

## Usage
1. **Run EDA**:
   - Open `EDA.ipynb` in Jupyter Notebook/JupyterLab to explore the dataset and visualize key insights.

2. **Train and Evaluate Models**:
   - Open `modeling.ipynb` to train SVM, Random Forest, and XGBoost models and compare their performance.
   - The notebook includes code to load the dataset, preprocess data, train models, and evaluate results.

3. **Use the Pre-trained Model**:
   - Load the saved Random Forest model (`RandomForest_model.joblib`) using the `joblib` library for predictions:
     ```python
     import joblib
     model = joblib.load('RandomForest_model.joblib')
     # Use model for predictions
     ```

## Requirements
The project dependencies are listed in `requirements.txt`. Key libraries include:
- `pandas`
- `numpy`
- `scikit-learn` (for SVM and Random Forest)
- `xgboost`
- `matplotlib` and `seaborn` (for visualization)
- `jupyter` (for running notebooks)

Install them using:
```bash
pip install -r requirements.txt
```

## Results
- The SVM model outperformed XGBoost and Random Forest in terms of classification performance.
- Detailed results, including metrics and comparisons, are available in `modeling.ipynb`.

## Future Work
- Explore additional feature engineering techniques to further improve model performance.
- Test other advanced algorithms (e.g., neural networks) for comparison.
- Deploy the Random Forest model in a production environment for real-time predictions.

## Contact
For questions or contributions, please contact [Ali] at [ali.sarafraz530@gmail.com].