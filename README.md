# Malnutrition Detection in South Sudanese Children: Machine Learning Optimization Study

## Data Source

* This project uses the Children Malnutrition Dataset from Kaggle: [https://www.kaggle.com/datasets/albertkingstone/children-malnutrition-dataset](https://www.kaggle.com/datasets/albertkingstone/children-malnutrition-dataset)

## Problem Definition

Child malnutrition in South Sudan presents a critical public health challenge. This study applies machine learning and deep learning techniques to classify and detect malnutrition in children under 5, based on anthropometric and health data. The goal is to identify optimal model configurations to maximize predictive performance and provide actionable insights.

---

## Model Architectures and Implementations

The following models were implemented and evaluated:

### 1. Classical ML Algorithm: SVM with Hyperparameter Tuning

* A Support Vector Machine was tuned using GridSearchCV with a search space covering `C`, `gamma`, and `kernel`.
* This model achieved strong performance with minimal overfitting.

### 2. Simple Neural Network (No Optimization)

* A baseline feedforward neural network with 4 hidden layers.
* No dropout, regularization, or early stopping was applied.
* Used SGD optimizer with a fixed learning rate of 0.01 for 5 epochs.

### 3. Optimized Neural Network (5 Distinct Instances)

Five neural network instances were trained using distinct combinations of optimizers, regularization, dropout, and early stopping:

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Dropout Rate | Accuracy | F1 Score | Precision | Recall | Val Loss |
| -------- | --------- | ----------- | ------ | -------------- | ------ | ------------- | ------------ | -------- | -------- | --------- | ------ | -------- |
| 1        | SGD       | None        | 5      | No             | 4      | 0.01          | 0.0          | 0.8827   | 0.8530   | 0.8264    | 0.8827 | 0.3089   |
| 2        | Adam      | L2          | 5      | Yes            | 5      | 0.09          | 0.2          | 0.8840   | 0.8548   | 0.8294    | 0.8840 | 0.5028   |
| 3        | RMSprop   | L1          | 5      | Yes            | 5      | 0.09          | 0.3          | 0.8547   | 0.8339   | 0.8317    | 0.8547 | 2.3466   |
| 5        | RMSprop   | None        | 5      | Yes            | 4      | 0.07          | 0.25         | 0.8307   | 0.8143   | 0.8266    | 0.8307 | 0.4176   |

### 4. XGBoost with Hyperparameter Tuning

* Tuned using a simplified grid, focusing on depth and learning rate.
* Delivered strong results, with performance close to the best neural networks.

---

## Evaluation Metrics

Models were evaluated using the following metrics on a held-out test set:

* **Accuracy:** Overall correctness.
* **F1-score (weighted):** Balance between precision and recall.
* **Precision (weighted):** Correct positive predictions.
* **Recall (weighted):** Correctly identified positives.
* **Loss:** Training loss (only for neural networks).

---

## Comprehensive Error Analysis and Optimization Impact

### Instance 1 (SGD, No Regularization, No Dropout)

* **Pros:** Simple and fast to train; decent accuracy for a baseline.
* **Cons:** Lacks any form of regularization or early stopping, making it prone to overfitting despite the short training time.
* **Observation:** The low validation loss (0.3089) indicates a reasonable fit, but absence of control mechanisms may hinder generalization on more complex data.

### Instance 2 (Adam, L2 Regularization, Moderate Dropout, Early Stopping)

* **Pros:** Adam optimizer adapts learning rates during training, L2 regularization penalizes complex weights, and dropout (0.2) reduces reliance on specific neurons.
* **Cons:** Training stopped early due to validation loss plateauing, which might have limited maximum performance.
* **Observation:** Achieved best performance among this batch. Suggests that combining multiple moderate regularization techniques leads to good generalization without overfitting.

### Instance 3 (RMSprop, L1 Regularization, High Dropout, Early Stopping)

* **Pros:** L1 regularization encourages sparsity (simpler models), high dropout forces robust learning.
* **Cons:** Likely underfit due to aggressive regularization. High dropout (0.3) combined with L1 might have stripped away useful learning.
* **Observation:** Highest validation loss (2.3466) confirms poor learning capacity. Caution is needed when stacking multiple strong regularizers.

### Instance 5 (RMSprop, No Regularization, Moderate Dropout, Early Stopping)

* **Pros:** Balanced dropout (0.25) provided enough regularization without need for L1/L2. RMSprop handled non-stationary learning well.
* **Cons:** Slightly lower performance than others, but the model was relatively stable.
* **Observation:** Surprisingly effective without formal regularization. Shows that dropout alone, when tuned properly, can suffice as regularizer in certain architectures.

### Additional Notes:

* **Validation Loss as Indicator:** Models with overly high validation loss (like Instance 3) showed underfitting, while models with very low loss but lower accuracy (like Instance 5) indicate overfitting avoidance but possible capacity limits.
* **Effect of Optimizer:** Adam and RMSprop both handled learning rate dynamics well; SGD underperformed likely due to lack of momentum or adaptive mechanisms.
* **Layer Depth:** Deeper models (5 layers) seemed to benefit from proper regularization. Instance 1 had 4 layers and decent performance, but lacked fine control.

---

## Summary of Key Results

| Model                            | Accuracy | F1 Score | Precision | Recall |
| -------------------------------- | -------- | -------- | --------- | ------ |
| Tuned SVM                        | 0.9493   | 0.9486   | 0.9487    | 0.9493 |
| XGBoost (Tuned)                  | 0.9307   | 0.9293   | 0.9284    | 0.9307 |
| Simple NN                        | 0.8907   | 0.8619   | 0.8371    | 0.8907 |
| Optimized NN (Best - Instance 5) | 0.8307   | 0.8143   | 0.8266    | 0.8307 |

---

## Conclusion

The best performing model overall was the **Tuned SVM** with an accuracy of **94.93%**, closely followed by **Tuned XGBoost (93.07%)**. While neural networks performed adequately, especially with optimization, classical ML models proved more stable and generalized better on the malnutrition dataset.

This study demonstrates the importance of tailoring optimization strategies to dataset characteristics, and highlights how even simple regularization and training techniques can significantly affect model performance in sensitive health-related domains like malnutrition detection.
