# Parkinson’s Disease Detection using Voice Measurements

This project implements a complete machine learning pipeline to detect Parkinson’s Disease based on vocal biomarkers. Using biomedical voice measurements from individuals with and without Parkinson’s, I have built, trained, and evaluated classification models to predict the presence of the disease.

---

## Objective

To develop a reliable ML-based diagnostic tool that identifies Parkinson’s Disease using voice data, demonstrating expertise in medical data preprocessing, feature selection, handling class imbalance, and supervised learning techniques.

---

## Dataset

- Source: GitHub 
- Features: Frequency, jitter, shimmer, noise-to-harmonics ratio (NHR), etc.
- Target: `status` (1 = Parkinson’s, 0 = Healthy)

---

## Tech Stack

- **Languages**: Python
- **Libraries**: NumPy, Pandas, Scikit-Learn, imbalanced-learn, Keras (TensorFlow backend)
- **Tools**: Google Colab, Matplotlib/Seaborn (for visualization)

---

## ML Pipeline Steps

1. **Data Loading & Inspection**  
   - Loaded and explored the dataset for structure and null values.

2. **Preprocessing & Feature Scaling**  
   - Cleaned data and optionally standardized feature ranges.

3. **Train-Test Split**  
   - Divided the data into training (80%) and test (20%) subsets.

4. **Feature Selection**  
   - Employed **Recursive Feature Elimination (RFE)** to select the most informative predictors.

5. **SMOTE Oversampling**  
   - Applied **Synthetic Minority Over-sampling Technique (SMOTE)** to balance class distribution.

6. **Model Training**  
   - Trained two models:
     - **K-Nearest Neighbors (KNN)**
     - **Feedforward Neural Network (FNN)** using Keras

7. **Hyperparameter Tuning**  
   - Utilized **RandomizedSearchCV** for tuning the KNN model.

8. **Performance Evaluation**  
   - Evaluated models using:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - Confusion Matrix

9. **Result Export**  
   - Saved processed data and results to CSV for downstream use.

---

## Results Summary

| Model | Accuracy | Precision | F1 Score |
|-------|----------|-----------|----------|
| KNN   | 0.8718   | 0.96154   | 0.9091   |
| FNN   | 0.8974   | 0.9629    | 0.9286  |

---

## Impact

- Showcases real-world application of ML in healthcare.
- Acquired end-to-end understanding of:
  - Supervised learning
  - Feature engineering
  - Data balancing (SMOTE)
  - Neural networks (FNN) and traditional ML (KNN)


