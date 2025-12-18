# Toxic Comment Classification

Course project for CS178 Machine Learning and Data Mining at UCI. We developed, validated, and compared 3 classification models for identifying toxic comments.

# Overview

This project is a multi-label problem in identifying if any toxicity labels apply to a given comment.

- **Data**: Comments from Wikipedia talk page. Part of Jigsaw Toxic Comment Classification Challenge on Kaggle
- **Labels**: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate. Zero or more could apply to a commment.
- **Models**: Logistic Regression, Linear SVM, DistilRoBERTa
- **Methods**: For each model, we conduct small grid search over hyperparamters of the highest importance, using 4-fold CV

## Model Details

| Model               | Description                            |
| ------------------- | -------------------------------------- |
| Logistic Regression | TF-IDF + OneVsRest Logistic Regression |
| Linear SVM          | TF-IDF + OneVsRest LinearSVC           |
| DistilRoBERTa       | Fine-tuned `distilroberta-base`        |

## Grid Search Parameters

| Model               | Parameters                                         |
| ------------------- | -------------------------------------------------- |
| Logistic Regression | `max_features`: [20k, 30k, 40k], `C`: [1, 10, 100] |
| Linear SVM          | `max_features`: [10k, 20k, 30k], `C`: [0.1, 1, 10] |
| DistilRoBERTa       | `lr`: [1e-5, 2e-5, 5e-5], `batch_size`: [64, 128]  |

## Best Configuration

| Model               | Best Parameters               |
| ------------------- | ----------------------------- |
| Logistic Regression | `max_features`=40000, `C`=10  |
| Linear SVM          | `max_features`=30000, `C`=1.0 |
| DistilRoBERTa       | `lr`=5e-5, `batch_size`=64    |

## Data

### Training and test dataset after preprocessing

| Data  | Samples | Labels |
| ----- | ------- | ------ |
| Train | 159,571 | 6      |
| Test  | 63,978  | 6      |

### Label Distribution

| Label         | Positive Samples | Percentage |
| ------------- | ---------------- | ---------- |
| toxic         | 15,294           | 9.58%      |
| obscene       | 8,449            | 5.29%      |
| insult        | 7,877            | 4.94%      |
| severe_toxic  | 1,595            | 1.00%      |
| identity_hate | 1,405            | 0.88%      |
| threat        | 478              | 0.30%      |

**Class Imbalance**: 89.83% of comments have no toxic labels, indicating severe class imbalance. 

## Project Structure

```
Toxic-Terminators/
├── data/
│   ├── test_1.csv                       # Test dataset (labeled)
│   ├── test_labels.csv                  # Test labels
│   ├── test.csv                         # Test dataset (unlabeled)
│   └── train.csv                        # Training dataset
├── models/
│   ├── logistic_regression.ipynb        # Logistic Regression training
│   ├── logistic_regression_tuning.ipynb # Hyperparameter tuning
│   ├── linear_svm.ipynb                 # Linear SVM training
│   ├── linear_svm_tuning.ipynb          # Hyperparameter tuning
│   ├── bert.ipynb                       # DistilRoBERTa training
│   └── bert_tuning.ipynb                # Hyperparameter tuning
├── scripts/
│   ├── eda.ipynb                        # Exploratory data analysis
│   ├── preprocess.ipynb                 # Data preprocessing
│   └── compare.ipynb                    # Model comparison and visualization
└── results/
    ├── logistic_regression_results.json # LR evaluation metrics
    ├── linear_svm_results.json          # SVM evaluation metrics
    ├── bert_results.json                # DistilRoBERTa evaluation metrics
    └── *_tuning_*.json                  # Hyperparameter tuning results
```

## Results

### Model Performance Comparison

| Model               | Macro Precision | Macro Recall | Macro F1  | Macro AUC-PR |
| ------------------- | --------------- | ------------ | --------- | ------------ |
| Logistic Regression | 0.347           | **0.820**    | 0.477     | 0.576        |
| Linear SVM          | 0.371           | 0.735        | 0.485     | 0.546        |
| DistilRoBERTa       | **0.543**       | 0.692        | **0.604** | **0.630**    |

### Per-Label AUC-PR for best model (DistilRoBERTa)

| Label         | AUC-PR |
| ------------- | ------ |
| toxic         | 0.782  |
| severe_toxic  | 0.329  |
| obscene       | 0.794  |
| threat        | 0.479  |
| insult        | 0.761  |
| identity_hate | 0.633  |

## Key Findings

**1. Precision-Recall Trade-off**: Logistic Regression achieves the highest recall (0.820) but lowest precision (0.347), predicting toxic labels more liberally. With TF-IDF max_features=40k, the model weight term features more heavily, resulting in high sensitivity thus many false positives. The linear decision boundary cannot capture complex interactions between terms, leading to overgeneralization. DistilRoBERTa balances precision (0.543) and recall (0.692) better through contextual embeddings.

**2. DistilRoBERTa Dominance**: The transformer-based model outperforms traditional ML approaches across F1 and AUC-PR metrics. Its bidirectional attention mechanism captures nuanced context, textual relationships, and implicit toxicity that sparse TF-IDF features miss. DistilRoBERTa achieves 56% higher precision than Logistic Regression (0.543 vs 0.347), suggesting better classification of borderline cases where context matters.

**3. Rare Label Challenge**: Severe_toxic (0.329 AUC-PR) and threat (0.479 AUC-PR) show significantly lower performance across all models because these labels suffer from insufficient training examples. This is particularly problematic for DistilRoBERTa which requires more data to learn effective representations. The macro-averaged metrics are heavily impacted by these weak labels.

**4. Computational vs Performance Trade-off**: While DistilRoBERTa requires GPU training (>1 hour) and significantly slower inference (>20 min vs <2 min for traditional ML models), the substantial gains in F1 (0.604 vs 0.485) and AUC-PR (0.630 vs 0.576) could justify the computational cost, if classification quality is prioritized over latency.

## Oversight

To address the severe class inbalance, we used class-weighted loss for training logistic regression and linear SVM ```class_weight = "balanced"```, but we did not do that for DistilRoBERTa ```loss = BCEWithLogitsLoss(pos_weight=class_ratio)```. if implemented, this should allow the transformer model to perform even better over classical ML baselines. 

Furthermore, post-training threshold tuning ```threshold[label] = argmax_F1(precision_recall_curve)``` could also improve classification power by effectively raising recall for rare classes, our toxic labels. 


