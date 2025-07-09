import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, roc_curve,log_loss
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the dataset
df = pd.read_csv(r'Loan_Approval_Prediction.csv') #File path to your dataset

# Data Preprocessing
X_loan = df.drop(['loan_id', 'loan_status'], axis=1)
y_loan = df['loan_status']
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_loan, y_loan, test_size=0.2, random_state=42)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in X_train_l.select_dtypes(include=['object']).columns:
    X_train_l[column] = label_encoder.fit_transform(X_train_l[column])
    X_test_l[column] = label_encoder.transform(X_test_l[column])
# Standardize the features
scaler = StandardScaler()
X_train_l = scaler.fit_transform(X_train_l)
X_test_l = scaler.transform(X_test_l)
# Logistic Regression Classifier
log_reg = LogisticRegression(random_state=42, max_iter=1000)
# Fit the model
log_reg.fit(X_train_l, y_train_l)
# Predictions
y_pred_l = log_reg.predict(X_test_l)
# Evaluation Metrics
accuracy_l = accuracy_score(y_test_l, y_pred_l)
precision_l = precision_score(y_test_l, y_pred_l, pos_label='Approved')
recall_l = recall_score(y_test_l, y_pred_l, pos_label='Approved')
f1_l = f1_score(y_test_l, y_pred_l, pos_label='Approved')
log_loss_l = log_loss(y_test_l, log_reg.predict_proba(X_test_l))
# Print evaluation metrics
print(f"Accuracy: {accuracy_l * 100:.2f}%")
print(f"Precision: {precision_l * 100:.2f}%")
print(f"Recall: {recall_l * 100:.2f}%")
print(f"F1 Score: {f1_l * 100:.2f}%")
print(f"Log Loss: {log_loss_l:.2f}")
# Confusion Matrix
cm_l = confusion_matrix(y_test_l, y_pred_l, labels=['Approved', 'Rejected'])
plt.figure(figsize=(8, 6))
sns.heatmap(cm_l, annot=True, fmt='d', cmap='Blues', xticklabels=['Approved', 'Rejected'], yticklabels=['Approved', 'Rejected'])
plt.title('Confusion Matrix for Loan Approval Prediction')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC Curve
y_prob_l = log_reg.predict_proba(X_test_l)[:, 1]
fpr_l, tpr_l, thresholds_l = roc_curve(y_test_l, y_prob_l, pos_label='Approved')
roc_auc_l = roc_auc_score(y_test_l, y_prob_l)
plt.figure(figsize=(8, 6))
plt.plot(fpr_l, tpr_l, color='blue', label=f'ROC Curve (AUC = {roc_auc_l:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid()
plt.show()