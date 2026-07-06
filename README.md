# Credit Card Fraud Detection

A machine learning project to detect fraudulent credit card transactions using the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle.

## Problem Statement

Can we build a model that reliably catches fraudulent transactions despite them being extremely rare, without generating excessive false alarms?

The dataset contains 284,807 European credit card transactions over two days, with only 492 labeled as fraud (about 0.17% of the data). Features V1 to V28 are PCA-transformed and anonymized. Time and Amount are the only raw fields, alongside the target Class (0 = normal, 1 = fraud).

Accuracy is not a meaningful metric here. A model that predicts "Not Fraud" for every transaction scores about 99.8% accuracy while catching zero fraud. This project focuses on the precision-recall tradeoff instead: missing fraud costs money, while flagging too many legitimate transactions creates unnecessary manual review work and erodes customer trust.

## Approach

1. Exploratory data analysis: class distribution, transaction amount and time patterns, correlation with the target.
2. Preprocessing: null and duplicate checks, log-transform and scaling of Amount and Time, stratified train/test split.
3. Class imbalance handling: class weighting, SMOTE, and undersampling, compared against each other.
4. Models trained: Logistic Regression, Decision Tree, Random Forest, XGBoost, KNN, Naive Bayes.
5. Evaluation using Precision, Recall, F1-score, and confusion matrices rather than accuracy alone.
6. Threshold tuning on the final model to adjust the precision-recall tradeoff.
7. Feature importance analysis.

## Results

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | Accuracy |
|---|---|---|---|---|
| Logistic Regression + class weight | 0.56 | 0.82 | 0.67 | 0.9985 |
| Logistic Regression + undersampling | 0.06 | 0.88 | 0.11 | 0.9750 |
| Logistic Regression + SMOTE | 0.06 | 0.86 | 0.11 | 0.9760 |
| XGBoost | 0.95 | 0.71 | 0.81 | 0.9994 |
| Decision Tree | 0.86 | 0.71 | 0.78 | 0.9993 |
| Random Forest | 0.95 | 0.76 | 0.84 | 0.9995 |
| KNN | 0.91 | 0.70 | 0.79 | 0.9994 |
| Naive Bayes | 0.06 | 0.80 | 0.11 | 0.9780 |

Aggressive rebalancing methods (SMOTE, undersampling, Naive Bayes) achieve high recall but collapse to a precision of about 0.06, meaning roughly 94% of their fraud alerts are false alarms. These are not usable in practice despite the high recall.

**Final model: Random Forest.** It achieves the best F1-score (0.84) with high precision (0.95), solid recall (0.76), and a small train-test performance gap, indicating good generalization rather than overfitting.

### Threshold Tuning

The default classification threshold of 0.5 was compared against alternative thresholds:

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.2 | 0.77 | 0.82 | 0.79 |
| 0.3 | 0.83 | 0.80 | 0.82 |
| 0.4 | 0.90 | 0.77 | 0.83 |
| 0.5 (default) | 0.95 | 0.76 | 0.84 |
| 0.6 | 0.98 | 0.69 | 0.81 |

Since missing a fraudulent transaction is generally costlier than a false alarm, a threshold of 0.2 was selected for deployment. This improves recall from 0.76 to 0.82 at a moderate precision cost (0.95 to 0.77).

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