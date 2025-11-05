## Credit Based Loan Prediction

A machine learning project that predicts loan approval status for applicants based on demographic, financial, and employment data. It demonstrates end-to-end ML workflow including data cleaning, feature engineering, model training, evaluation, and generating test predictions. Multiple algorithms are compared to select the best-performing model, and the notebook is ready to run in Google Colab or Jupyter.

## Dataset
The dataset contains information about loan applicants, including demographic, financial, and employment details.


**Key columns:**
- `Gender` – Male/Female
- `Married` – Applicant married or not
- `Dependents` – Number of dependents
- `Education` – Graduate/Not Graduate
- `Self_Employed` – Self-employed or not
- `ApplicantIncome` – Applicant’s income
- `CoapplicantIncome` – Co-applicant’s income
- `LoanAmount` – Loan amount requested
- `Loan_Amount_Term` – Term of the loan in months
- `Credit_History` – Credit history meets guidelines
- `Property_Area` – Urban/Semiurban/Rural
- `Loan_Status` – Target variable (Y/N)

##  Libraries Used
- pandas, numpy
- scikit-learn
- matplotlib/seaborn (optional for EDA)

## Workflow

### 1. Data Preprocessing
- Handle missing values:
  - Numeric columns filled with mean
  - Categorical columns filled with mode
- Outlier treatment: Log transformation of `Loan_Amount_Term`
- Feature engineering: Added `TotalIncome = ApplicantIncome + CoapplicantIncome`
- One-hot encoding for categorical variables
- Train/test feature alignment

### 2. Train/Test Split
- 80% training, 20% cross-validation

### 3. Feature Scaling
- Standard scaling applied to numeric features for models sensitive to magnitude (Logistic Regression, SVM, KNN)

### 4. Model Training & Evaluation
Models trained:
- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- Naive Bayes
- K-Nearest Neighbors (KNN)
- Gradient Boosting Classifier  

**Evaluation:** Cross-validation accuracy and confusion matrix for each model. Best model selected automatically based on accuracy.

### 5. Test Predictions
- Best model retrained on full training data
- Predictions made on test set
- Results saved as `Credit_Predictions.csv` including `Loan_ID`


## Results
- Baseline model accuracy (CV):
  - Logistic Regression: ~78–79%
  - Naive Bayes: ~78%
  - Random Forest: ~77%
- Best model chosen for final predictions based on highest accuracy.

