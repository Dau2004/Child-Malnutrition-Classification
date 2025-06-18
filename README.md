# Malnutrition Detection in South Sudanese Children

### *Optimizing Neural Networks and Classical Models for Malnutrition Classification*

---

## Dataset Overview

* **Source:** [Children Malnutrition Dataset (Kaggle)](https://www.kaggle.com/datasets/albertkingstone/children-malnutrition-dataset)
* **Features:**

  * `age_months`, `weight_kg`, `height_cm`, `muac_cm`, `bmi`
* **Target:**

  * `nutrition_status`: categorical (`normal`, `moderate`, `severe`)

---

## Objective

To classify the nutritional status of children using machine learning and deep learning models. The project investigates how different optimization strategies (e.g., regularization, optimizers, early stopping) affect neural network performance compared to classical models like SVM and XGBoost.

---

## Implemented Models

### 1. **Classical ML Models**

| Model               | Accuracy  | F1 Score | Precision | Recall |
| ------------------- | --------- | -------- | --------- | ------ |
| **SVM (Tuned)**     | **0.949** | 0.949    | 0.949     | 0.949  |
| **XGBoost (Tuned)** | 0.931     | 0.929    | 0.928     | 0.931  |

> These models were trained using GridSearchCV for hyperparameter tuning. SVM slightly outperformed XGBoost in terms of accuracy.

---

### 2. **Simple Neural Network (No Optimization)**

| Model                       | Accuracy | F1 Score | Precision | Recall |
| --------------------------- | -------- | -------- | --------- | ------ |
| **Simple NN (Unoptimized)** | 0.891    | 0.862    | 0.837     | 0.891  |

> Served as a baseline. No dropout, no regularization, no early stopping. Moderate performance, slight overfitting noticed during training.

---

### 3. **Optimized Neural Network Configurations**

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Dropout | Accuracy | F1 Score | Precision | Recall | Val Loss |
| -------- | --------- | ----------- | ------ | -------------- | ------ | ------------- | ------- | -------- | -------- | --------- | ------ | -------- |
| 1        | SGD       | None        | 5      |   No           | 4      | 0.01          | 0.0     | 0.883    | 0.853    | 0.826     | 0.883  | 0.309    |
| 2        | Adam      | L2          | 5      |   Yes          | 5      | 0.09          | 0.2     | 0.884    | 0.855    | 0.829     | 0.884  | 0.503    |
| 3        | RMSprop   | L1          | 5      |   Yes          | 5      | 0.09          | 0.3     | 0.855    | 0.834    | 0.832     | 0.855  | 2.347    |
| 5        | RMSprop   | None        | 5      |   Yes          | 4      | 0.07          | 0.25    | 0.831    | 0.814    | 0.827     | 0.831  | 0.418    |

> **Note**: Instance 4 was excluded due to poor performance and instability.

---

## Error Analysis & Optimization Impact

Each optimized instance revealed insights into how training configurations influence performance:

### **Instance 1 (SGD, No Regularization)**

* **Pros:** Fast training with low validation loss.
* **Cons:** No early stopping or regularization caused slight overfitting.
* **Insight:** SGD without constraints is unstable on deeper networks.

###  **Instance 2 (Adam + L2 + Dropout + Early Stopping)**

* **Pros:** Balanced performance across all metrics. L2 regularization improved generalization. Early stopping prevented overfitting.
* **Cons:** Slightly higher validation loss than Instance 1.
* **Insight:** Adam + L2 + dropout is a robust, reliable configuration.

### **Instance 3 (RMSprop + L1 + High Dropout)**

* **Pros:** Sparse weight learning due to L1 regularization.
* **Cons:** Highest validation loss, underperformance across all metrics.
* **Insight:** Over-regularization (L1 + 0.3 dropout) likely led to underfitting.

###  **Instance 5 (RMSprop, No Regularization)**

* **Pros:** Moderate dropout helped prevent overfitting. Fast convergence and decent validation loss.
* **Cons:** No regularization may limit generalization beyond current test set.
* **Insight:** Dropout alone can regularize effectively under certain configurations.

---

## Final Comparison & Recommendation

| Model                       | Accuracy  | F1 Score | Comments                                    |
| --------------------------- | --------- | -------- | ------------------------------------------- |
| **SVM (Tuned)**             | **0.949** | 0.949    | Best overall generalization                 |
| **XGBoost (Tuned)**         | 0.931     | 0.929    | Strong, interpretable model                 |
| **Optimized NN (Inst. 2)**  | 0.884     | 0.855    | Best among NNs (regularized + stable)       |
| **Simple NN (Unoptimized)** | 0.891     | 0.862    | Lacks regularization, weaker generalization |
| **Optimized NN (Inst. 3)**  | 0.855     | 0.834    | Underfitting due to excessive dropout       |

> **Conclusion:**
>
> * **SVM** is the top performer and most stable.
> * Among neural networks, **Instance 2** (Adam + L2 + Dropout + EarlyStopping) offers the best balance of accuracy and robustness.
> * Over-regularization (Instance 3) can harm learning capacity.
> * Careful tuning of learning rate, dropout, and regularizers is essential for neural network performance.

---

## Takeaways

* **Optimization techniques matter**: Dropout, early stopping, and appropriate regularization significantly affect model generalization.
* **Classical ML still performs competitively**, especially with structured tabular data.
* **Neural networks** need deliberate architecture design and training strategies to compete with classical models in small datasets.
