# **Malnutrition Detection in South Sudanese Children**  
### *Machine Learning Optimization Study*

**Author:** Chol Daniel  
**Course:** Introduction to Machine Learning  
**Institution:** African Leadership University  

📺 **Video Presentation:**  
https://youtu.be/S8qr4t3lIqI  

---

## **📌 Project Overview**

Child malnutrition remains one of South Sudan’s most urgent health challenges, where delayed diagnosis and limited diagnostic tools place thousands of children at risk. This project explores how **machine learning and deep learning** models can support earlier detection of malnutrition using standard anthropometric measurements.

The project reflects my broader mission to develop AI-driven decision-support tools that strengthen health systems in low-resource settings.

---

## **📂 Dataset**

**Source:** Children Malnutrition Dataset (Kaggle)  
https://www.kaggle.com/datasets/albertkingstone/children-malnutrition-dataset  

### Features:
- `age_months`
- `weight_kg`
- `height_cm`
- `muac_cm` (Mid–Upper Arm Circumference)
- `bmi`
- `nutrition_status` (normal, moderate, severe)

These features match WHO standards for malnutrition classification.

---

## **🎯 Problem Definition**

**Objective:**  
Build and optimize multiple ML and DL models to classify children into *normal*, *moderate*, or *severe* malnutrition categories.

**Why it matters:**  
Many health workers in South Sudan operate without digital tools that support fast, accurate assessment. By comparing classical ML models with deep neural networks, this study identifies which approaches offer reliable performance suitable for real-world deployment.

---

## **🧠 Model Architectures Implemented**

### **1. Support Vector Machine (SVM) — Tuned**
- Tuned using GridSearchCV (`C`, `gamma`, `kernel`)  
- Most accurate and stable model  
- **Accuracy: 94.93%**

### **2. Baseline Neural Network**
- 4 hidden layers  
- SGD optimizer  
- Baseline benchmark before optimization  

### **3. Optimized Neural Networks (5 Instances)**  
Each instance varied in:
- Optimizer (Adam, RMSprop, SGD)
- L1/L2 regularization  
- Dropout (0.2–0.3)  
- Learning rate  
- Early stopping  
- Model depth  

This iterative experimentation mirrors my learning process—adjusting regularization, understanding underfitting, and tuning for stability.

### **4. XGBoost — Tuned**
- Tuned depth + learning rate  
- **Accuracy: 93.07%**

---

## **📊 Evaluation Metrics**

Models evaluated on a held-out test set using:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Validation Loss (for NN models)  

These metrics evaluate both correctness and real-world reliability.

---

## **🔍 Optimization Insights**

### **Best Classical Model:**  
✅ **Tuned SVM**  
- Most stable across experiments  
- Best accuracy and generalization  
- Suitable for low-resource deployments

### **Best Neural Network:**  
⚖️ **Instance with balanced dropout (0.25) & RMSprop**  
- Good stability  
- Lower risk of overfitting  
- Moderate complexity

### **Key Lessons**
- Over-regularization (e.g., high dropout + L1) leads to underfitting  
- Neural networks require careful tuning to match simpler models  
- Classical ML (SVM/XGBoost) outperformed deep networks on this dataset  
- Validation loss patterns were crucial for diagnosing under/overfitting  

---

## **🏆 Results Summary**

| Model                            | Accuracy | F1 Score |
| -------------------------------- | -------- | -------- |
| **Tuned SVM**                    | **0.9493** | **0.9486** |
| XGBoost (Tuned)                  | 0.9307   | 0.9293   |
| Simple Neural Network            | 0.8907   | 0.8619   |
| Optimized NN (Best Instance)     | 0.8307   | 0.8143   |

---

## **📌 Conclusion**

This study demonstrates that **classical ML approaches outperform deep learning models** for this malnutrition dataset. Their stability, simplicity, and high accuracy make them ideal for real-world diagnostic support tools in South Sudan.

The insights gained from tuning, evaluating, and analyzing these models directly shaped my long-term direction:  
→ building AI-powered decision-support tools for health workers in low-resource environments.

---

## **🛠️ How to Run This Project**

### **1. Clone the Repository**
```bash
git clone https://github.com/your-username/malnutrition-detection-ml.git
cd malnutrition-detection-ml
