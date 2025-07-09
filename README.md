###Loan Approval Prediction
This repository contains machine learning models built using Python to predict loan approval status based on applicant data. It includes implementations of Decision Tree, Logistic Regression, and Random Forest classifiers.

📁 Files Included
Loan_Approval_Prediction.csv: Dataset containing loan application records.

decision_tree_loan_approval_prediction.py: Implementation using Decision Tree Classifier.

logistic_regression_Loan_Approval_Prediction.py: Implementation using Logistic Regression.

random_forest_loan_approval_prediction.py: Implementation using Random Forest Classifier.

📊 Dataset Description
The dataset includes the following features:

loan_id: Unique loan identifier

applicant_income, coapplicant_income, loan_amount, loan_term: Numerical features

gender, married, dependents, education, self_employed, property_area, etc.: Categorical features

loan_status: Target variable (Approved, Rejected)

🧠 Models Used
Decision Tree

Categorical encoding via LabelEncoder

Evaluation using accuracy, precision, recall, F1 score, log loss, confusion matrix

Visualizes confusion matrix and tree

Logistic Regression

Encoding + feature standardization (StandardScaler)

Metrics: Accuracy, Precision, Recall, F1 Score, Log Loss

Includes ROC Curve

Random Forest

Feature importance plot

Visualization of a single decision tree from the ensemble

🔧 How to Run
Clone the repository:

bash
Copy
Edit
git clone https://github.com/your-username/loan-approval-prediction.git
cd loan-approval-prediction
Install the required libraries:

bash
Copy
Edit
pip install pandas numpy scikit-learn matplotlib seaborn
Run any script:

bash
Copy
Edit
python decision_tree_loan_approval_prediction.py
python logistic_regression_Loan_Approval_Prediction.py
python random_forest_loan_approval_prediction.py
📈 Output Examples
Accuracy, Precision, Recall, F1 Score

Confusion Matrix (heatmap)

ROC Curve (for Logistic Regression)

Feature Importance (for Random Forest)

Decision Tree visualizations

📌 Project Objective
The aim is to build a machine learning solution that helps financial institutions quickly determine the likelihood of loan approval, thereby improving operational efficiency and risk management.

✍️ Author
Dhruval Rana
Data Science | ML Projects | Python Developer
