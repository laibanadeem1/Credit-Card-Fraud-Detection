# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

## Problem Statement

Can we build a model that reliably catches fraudulent transactions despite them being extremely rare, without generating excessive false alarms?

The dataset contains 284,807 European credit card transactions over two days, with only 492 labeled as fraud (about 0.17% of the data). Features V1 to V28 are PCA-transformed and anonymized. Time and Amount are the only raw fields, alongside the target Class (0 = normal, 1 = fraud).

Accuracy is not a meaningful metric here. A model that predicts "Not Fraud" for every transaction scores about 99.8% accuracy while catching zero fraud. This project focuses on the precision-recall tradeoff instead: missing fraud costs money, while flagging too many legitimate transactions creates unnecessary manual review work and erodes customer trust.

## Approach

1. Exploratory data analysis: class distribution, transaction amount and time patterns, correlation with the target.
2. Preprocessing: null and duplicate checks, log-transform and scaling of Amount and Time (required for Logistic Regression and KNN), stratified 70/30 train/test split. Duplicate rows were deliberately not dropped, since a portion of them are fraud cases and removing them would shrink an already small minority class.
3. Class imbalance handling: class weighting, SMOTE, undersampling, and ADASYN.
4. Models trained: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Naive Bayes.
5. Evaluation using Precision, Recall, F1, F2, and confusion matrices rather than accuracy alone. F2 is reported alongside F1 since it weights recall more heavily, matching the project's priority of catching fraud over minimizing false alarms.
6. Overfitting checks via train/test performance comparison for every tree-based model.
7. Threshold tuning on the final model to adjust the precision-recall tradeoff.
8. Error analysis on false negatives: comparing feature patterns between missed fraud cases and correctly caught fraud cases.

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | Accuracy |
|---|---|---|---|---|
| Logistic Regression + class weight | 0.56 | 0.82 | 0.67 | 0.9985 |
| Logistic Regression + undersampling | 0.06 | 0.88 | 0.11 | 0.9750 |
| Logistic Regression + SMOTE | 0.06 | 0.86 | 0.11 | 0.9760 |
| XGBoost | 0.95 | 0.71 | 0.81 | 0.9994 |
| Decision Tree | 0.86 | 0.71 | 0.78 | 0.9993 |
| Random Forest (default threshold) | 0.95 | 0.76 | 0.84 | 0.9995 |
| KNN | 0.91 | 0.70 | 0.79 | 0.9994 |
| Naive Bayes | 0.06 | 0.80 | 0.11 | 0.9780 |

Aggressive rebalancing methods (SMOTE, undersampling, Naive Bayes) achieve high recall but collapse to a precision of about 0.06, meaning roughly 94% of their fraud alerts are false alarms. These are not usable in practice despite the high recall.

A hyperparameter search (RandomizedSearchCV) was also run on XGBoost. It reached a slightly higher test F1 (0.83) and F2 (0.81), but showed a large train-test performance gap (train F1 of 1.00 versus test F1 of 0.83), indicating overfitting. This result was excluded from final model selection for that reason, and is noted here as a check performed rather than a result relied on.

**Final model: Random Forest, evaluated at a tuned threshold of 0.2** (rather than the default 0.5). Random Forest was chosen over XGBoost for its better precision-recall balance and a smaller, more consistent train-test gap across every threshold tested.

### Threshold Tuning

The default classification threshold of 0.5 was compared against alternative thresholds:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.2 | 0.77 | 0.82 | 0.79 |
| 0.3 | 0.83 | 0.80 | 0.82 |
| 0.4 | 0.90 | 0.77 | 0.83 |
| 0.5 (default) | 0.95 | 0.76 | 0.84 |
| 0.6 | 0.98 | 0.69 | 0.81 |

Since missing a fraudulent transaction is generally costlier than a false alarm, F2 (which weights recall more heavily than F1) was used as the deciding metric rather than F1. A threshold of 0.2 was selected, giving an F2-score of 0.81 on the test set versus 0.79 at the default threshold. This improves recall from 0.76 to 0.82 at a moderate precision cost (0.95 to 0.77).

Train-set performance at the same threshold (precision 0.86, recall 0.85, F1 0.86, F2 0.85) is close to the test-set result, confirming the model generalizes well rather than overfitting to the training data.

### Additional Experiments

Undersampling and ADASYN were also tested as alternative imbalance-handling strategies. Both showed the same pattern as SMOTE: high recall but impractically low precision, and were not selected for the final model. A false-negative analysis was also performed, comparing feature averages between missed fraud cases and correctly identified fraud cases, to understand what the model consistently struggles to detect.

## Limitations

- Features V1 to V28 are anonymized PCA components, which limits interpretability of what specifically drives a fraud prediction.
- The dataset covers only two days of transactions from 2013 European cardholders. Patterns may not generalize to other time periods or regions.

## Project Structure

```
.
├── notebook.ipynb        # Full analysis: EDA, preprocessing, modeling, evaluation
├── app.py                 # Streamlit demo
├── requirements.txt        # Python dependencies
└── README.md
```

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