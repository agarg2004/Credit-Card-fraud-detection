# Credit-Card-fraud-detection
This project is aimed at classifying a transaction as fraud or not. The dataset is imbalanced, so resampling techniques have been applied to improve model performance. Various models have been implemented to increase the accuracy and recall(True negative rate), minimising false positive error.


# Dataset
The dataset used is "https://www.kaggle.com/datasets/arslanali4343/credit-card-cheating-detection-cccd/data" available on Kaggle. It contains transactions made by European cardholders in September 2013.

- **Number of rows:** 284,807
- **Number of fraud cases:** 492 (0.172%)
- **Feature count:** 30 anonymized features (V1–V28, Time, Amount) + 'Class' (target)


## Technologies Used

- Python 3
- Pandas
- NumPy
- scikit-learn
- Matplotlib / Seaborn (for visualization)

## ML Workflow

1. **Load and preprocess data**
   - Handle imbalanced classes using resampling (SMOTE)

2. **Train-Test Split**
   - Split the resampled dataset into training and testing sets (80/20 split).

3. **Model Training**
   - Train a K-Nearest Neighbors classifier using `KNeighborsClassifier`.
   - Train a Decision Tree classifiier
   - Train a random forest classifier

4. **Model Evaluation**
   - Evaluate performance using:
     - Confusion Matrix
     - Classification Report (Precision, Recall, F1-score)
     - Accuracy Score

## 📊 Results

Example output from the model:

