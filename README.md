# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

## Problem Statement

Can we build a model that reliably catches fraudulent transactions despite them being extremely rare, without generating excessive false alarms?

The dataset contains 284,807 European credit card transactions over two days, with only 492 labeled as fraud (about 0.17% of the data). Features V1 to V28 are PCA-transformed and anonymized. Time and Amount are the only raw fields, alongside the target Class (0 = normal, 1 = fraud).

Accuracy is not a meaningful metric here. A model that predicts "Not Fraud" for every transaction scores about 99.8% accuracy while catching zero fraud. This project focuses on the precision-recall tradeoff instead: missing fraud costs money, while flagging too many legitimate transactions creates unnecessary manual review work and erodes customer trust.

## Approach

1. Exploratory data analysis: class distribution, transaction amount and time patterns, correlation with the target.
2. Preprocessing: null and duplicate checks, RobustScaler applied to Time and Amount (chosen over StandardScaler for its lower sensitivity to outlier transactions), with V1-V28 passed through unchanged. All preprocessing is wrapped in a single `ColumnTransformer`, reused across every model via `Pipeline`, so every model is trained and evaluated on identical, consistently preprocessed data. Duplicate rows were deliberately not dropped, since a portion of them are fraud cases and removing them would shrink an already small minority class.
3. Data split: stratified 80/20 train/test split, with a further 75/25 split of the training data into sub-training/validation sets, used specifically for threshold tuning without ever exposing the test set during tuning.
4. Class imbalance handling: class weighting, SMOTE, undersampling, and ADASYN, compared against each other.
5. Models trained: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Naive Bayes.
6. Evaluation using Precision, Recall, F1, F2, and confusion matrices rather than accuracy alone. F2 is reported alongside F1 since it weights recall more heavily, matching the project's priority of catching fraud over minimizing false alarms.
7. Overfitting checks via train/test performance comparison, always matched to the same decision threshold, for every candidate final model.
8. Threshold tuning performed on a held-out validation split (not the test set), to avoid tuning against the same data used for final evaluation.
9. Error analysis on false negatives: comparing feature patterns between missed fraud cases and correctly caught fraud cases.

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | F2 (Fraud) |
|---|---|---|---|---|
| Logistic Regression + class weight | 0.56 | 0.82 | 0.67 | - |
| SMOTE / undersampling / Naive Bayes | ~0.06-0.09 | 0.80-0.88 | ~0.11 | - |
| XGBoost (default threshold) | 0.64 | 0.86 | 0.73 | - |
| XGBoost (validation-tuned threshold = 0.90) | 0.85 | 0.83 | 0.84 | 0.83 |
| Decision Tree | 0.86 | 0.71 | 0.78 | - |
| Random Forest (default threshold) | 0.92 | 0.81 | 0.86 | - |
| **Random Forest (validation-tuned threshold = 0.12)** | **0.69** | **0.88** | **0.77** | **0.83** |
| Random Forest + SMOTE | 0.83 | 0.82 | 0.82 | - |
| KNN | 0.91 | 0.70 | 0.79 | - |

Aggressive rebalancing methods (SMOTE, undersampling, Naive Bayes) achieve high recall but collapse to a precision in the range of 0.06-0.09, meaning the large majority of their fraud alerts are false alarms. These are not usable in practice despite the high recall.

Both XGBoost and Random Forest were tuned using a held-out validation split (not the test set) to select a classification threshold, then evaluated once on the untouched test set. On this basis, their F2-scores are close: XGBoost reaches 0.8316, Random Forest reaches 0.8333.

**Final model: Random Forest, evaluated at a validation-tuned threshold of 0.12.** Although XGBoost's F2-score was nearly identical, XGBoost showed a large train-test gap (train recall of 1.00 versus test recall of 0.83, a 17-point gap), indicating overfitting. This pattern appeared consistently across multiple XGBoost tuning attempts in this project, including an earlier hyperparameter search. Random Forest, evaluated at the same threshold on both train and test data, showed a much smaller gap (F2 of 0.846 on train versus 0.833 on test), confirming it generalizes reliably. Given comparable performance, the model with the smaller overfitting gap was selected.

### Threshold Tuning

Thresholds were selected using a validation split carved out of the training data (not the test set), evaluated across a range of candidate values using F2 as the deciding metric, since missing a fraudulent transaction is generally costlier than a false alarm:

| Threshold | Precision | Recall | F2 |
|---|---|---|---|
| 0.05 | 0.67 | 0.81 | 0.78 |
| 0.10 | 0.75 | 0.81 | 0.80 |
| 0.12 (selected) | 0.78 | 0.81 | 0.80 |
| 0.15 | 0.81 | 0.80 | 0.80 |
| 0.20 | 0.82 | 0.79 | 0.79 |
| 0.30 | 0.82 | 0.76 | 0.77 |
| 0.50 | 0.92 | 0.70 | 0.73 |

(Validation-set scores above; the final reported test-set performance at threshold 0.12 is precision 0.69, recall 0.88, F2 0.83.)

Threshold 0.12 was selected on the validation set and then evaluated once on the test set, improving recall to 0.88 at a real precision cost (0.69). Train-set performance at this same threshold (precision 0.79, recall 0.86, F2 0.846) is close to the test-set result (F2 0.833), confirming this threshold choice does not introduce overfitting.

### Additional Experiments

Undersampling and ADASYN were also tested as alternative imbalance-handling strategies. Both showed the same pattern as SMOTE: high recall but impractically low precision, and were not selected for the final model. A false-negative analysis was also performed, comparing feature averages between missed fraud cases and correctly identified fraud cases, to understand what the model consistently struggles to detect.

## Limitations

- Features V1 to V28 are anonymized PCA components, which limits interpretability of what specifically drives a fraud prediction. The final model uses 30 features in total: Time, Amount, and V1-V28.
- The dataset covers only two days of transactions from 2013 European cardholders. Patterns may not generalize to other time periods or regions.
- Error analysis showed that missed fraud cases (false negatives) are not near-miss transactions sitting just under the classification threshold; the model assigns them very low fraud probability with high confidence. These cases lack the extreme feature values that characterize easily detected fraud, suggesting a subtler fraud pattern that the available anonymized features may not fully capture. Attempting to address this with ADASYN, a resampling technique that targets hard-to-classify examples, did not improve results.

## Project Structure

```
.
├── notebook.ipynb              # Full analysis: EDA, preprocessing, modeling, evaluation
├── app.py                      # Streamlit demo
├── fraud_model.pkl             # Saved model: a single Pipeline containing both preprocessing and the trained Random Forest
├── sample_transactions.csv     # Synthetic sample file for trying the batch upload demo
├── requirements.txt            # Python dependencies
└── README.md
```

The saved model is a single scikit-learn `Pipeline` object, combining the RobustScaler preprocessing step and the trained Random Forest together. This means the app only needs to load one file and can pass raw transaction data (Time, Amount, V1-V28) directly to it, with no manual feature scaling required in the application code.

## Running the Project

Install dependencies:

```
pip install -r requirements.txt
```

Run the notebook to reproduce the analysis, or run the Streamlit demo:

```
streamlit run app.py
```

## Dataset

[Credit Card Fraud Detection, Kaggle (mlg-ulb)](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)