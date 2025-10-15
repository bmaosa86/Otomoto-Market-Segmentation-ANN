# Optimizing Otomoto’s Artificial Neural Network for Marketing Segmentation

**Author:** Benjamin Mong’are  

---

## 1. Project Overview
This project focuses on improving Otomoto’s customer segmentation by optimizing an Artificial Neural Network (ANN) model to better identify churn-risk customers.  
The goal was to use optimization algorithms to enhance model performance so that marketing teams can target the right customers for retention campaigns.

The analysis uses the **Telco Customer Churn dataset**, which contains customer demographics, subscriptions, contracts, and payment details.  
The objective is to classify customers as **likely to churn** or **likely to stay**, and understand which factors drive churn.

---

## 2. Tools and Libraries Used
- Python 3.11  
- TensorFlow / Keras  
- Scikit-learn  
- Pandas and NumPy  
- Matplotlib for visualizations  

All analysis was performed in **Jupyter Notebook**.

---

## 3. Model Development Process

### Data Preparation
- Converted `TotalCharges` to numeric and handled 11 missing values.  
- Mapped `Churn` to binary values (No = 0, Yes = 1).  
- Standardized numeric features and one-hot encoded categorical ones.

### Baseline Model
Built a simple ANN with two hidden layers (128 and 64 neurons) trained with early stopping and learning-rate reduction.

### Optimizer Comparison
Three optimizers were tested:
- **AdamW** — adaptive learning with weight decay  
- **RMSprop** — robust on noisy gradients  
- **SGD with Nesterov momentum** — strong generalization  

**RMSprop** achieved the best overall balance, with a **Macro F1 ≈ 0.72** and **ROC-AUC ≈ 0.84**.

### Threshold Tuning
Two thresholds were evaluated:
- **0.532 (max F1)** – balanced precision and recall  
- **0.446 (≈ 80 % recall)** – higher recall, more churners captured  

The final threshold was chosen based on marketing campaign priorities.

### Model Evaluation
Key metrics: Accuracy, F1, ROC-AUC, confusion matrix, ROC and PR curves, Lift and Gains charts.  
The optimized model captured most churners accurately while keeping false positives manageable.

### Feature Importance
Permutation importance showed that **tenure**, **contract type**, and **charges** were the strongest churn indicators.  
Customers on month-to-month contracts with higher charges were most likely to churn.

---

## 4. Files in This Repository
| File | Description |
|------|--------------|
| `Otomoto_ANN_Optimization.ipynb` | Main Jupyter Notebook with code, results, and discussion |
| `artifacts/best_model.keras` | Trained Keras model |
| `artifacts/preprocessor.pkl` | Fitted preprocessing pipeline |
| `artifacts/threshold.txt` | Selected decision threshold |
| `README.md` | Project summary |

---

## 5. How to Run
1. Install dependencies:  
   ```bash
   pip install tensorflow scikit-learn pandas matplotlib
