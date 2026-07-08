# ❤️ Heart Disease Prediction

## Problem Statement
Heart disease remains one of the leading causes of death worldwide, often progressing silently before a life-threatening event occurs. Many cases go undetected until it's too late.

This project analyzes clinical and diagnostic data — including chest pain patterns, blood pressure, cholesterol, and stress test results — to uncover patterns behind heart disease, with the goal of building a model that predicts a patient's risk early. Early detection like this can help influence doctors' decisions and prompt further testing or intervention before a critical event occurs.

## Dataset Used
This project uses the **Heart Disease Dataset (Comprehensive)** by Manu Siddhartha, sourced from IEEE DataPort:

📎 [Heart Disease Dataset (Comprehensive) — IEEE DataPort](https://ieee-dataport.org/open-access/heart-disease-dataset-comprehensive)
(also available on Kaggle as `heart-statlog-cleveland-hungary-final`)

> **Note:** This dataset is gated behind an IEEE DataPort / Kaggle login and must be downloaded manually.

## Analysis Process
During initial research, several data quality issues were identified and addressed before modeling:

- **Duplicate records** — ~272 duplicate rows found in the raw data
- **Placeholder zeros instead of true missing values** — `cholesterol` had ~172 zero values (~14.5% of rows), which are not physiologically possible and were treated as missing
- **Undocumented category in `st_slope`** — a value of `0` appears in the raw data despite documentation only describing categories 1, 2, and 3
- **Class balance** — checked across the dataset, since it merges 5 different source datasets that may not share identical positive/negative rates

These issues were addressed directly, followed by a full exploratory data analysis (EDA).

## Model Development
The following metrics were used to evaluate model performance:

- Balanced Accuracy
- F1 Score
- Precision
- Recall

These metrics were chosen specifically to ensure the model performs well across **both** classes, not just the majority class.

Three models of increasing complexity were compared using **Stratified K-Fold Cross-Validation**:

1. Logistic Regression
2. Random Forest Classifier
3. XGBoost Classifier

## Results
Initial evaluation revealed the model was favoring the `Male` class within the `Sex` column, a direct result of class imbalance in that feature (~76% male, ~24% female in the dataset). To address this, a `BalancedBaggingClassifier` was used to reduce this bias and improve performance across both groups.

## Key Insights and Recommendations

### Recommendations
Since the dataset contains just over 900 rows, a larger dataset is recommended to help the model learn richer feature interactions and generalize more reliably — particularly to improve performance for underrepresented subgroups (e.g., female patients).

### Potential Real-World Applications
Hospitals could use a model like this as a **supporting diagnostic tool**, helping doctors form and prioritize hypotheses during patient evaluation — not as a replacement for clinical judgment.

## How to Run the App
1. Clone or download the project folder.
2. Install the required dependencies:
```bash
   pip install -r requirements.txt
```
3. Run the app:
```bash
   python -m main
```
4. This starts a local web server. Open your browser and navigate to:
http://127.0.0.1:5000
5. Fill in the patient's clinical details in the form and click **"Predict"** to get the model's risk assessment.

Check out the deployed site [here](https://analystlab-africa-ml-projects.onrender.com/)
