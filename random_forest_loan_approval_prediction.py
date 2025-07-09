import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r'Loan_Approval_Prediction.csv') # File path to your dataset

# Data Preprocessing
X_loan = df.drop(['loan_id', 'loan_status'], axis=1)
y_loan = df['loan_status']

# Split the dataset into training and testing sets
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_loan, y_loan, test_size=0.2, random_state=42)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in X_train_l.select_dtypes(include=['object']).columns:
    X_train_l[column] = label_encoder.fit_transform(X_train_l[column])
    X_test_l[column] = label_encoder.transform(X_test_l[column])
# Random Forest Classifier
rf_clf = RandomForestClassifier(random_state=42)
# Fit the model
rf_clf.fit(X_train_l, y_train_l)
# Predictions
y_pred_l = rf_clf.predict(X_test_l)
# Evaluation Metrics
accuracy_l = rf_clf.score(X_test_l, y_test_l)
print(f"Accuracy: {accuracy_l * 100:.2f}%")
# Feature Importance
feature_importances = rf_clf.feature_importances_
feature_names = X_train_l.columns
# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
# Print feature importances
print("\nFeature Importances:")
print(importance_df)
# Plot feature importances
plt.figure(figsize=(10, 6))
importance_df.plot(kind='bar', x='Feature', y='Importance', legend=False)
plt.title('Feature Importances in Random Forest Classifier')
plt.xlabel('Features')
plt.ylabel('Importance')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Visualize one of the trees in the Random Forest
plt.figure(figsize=(20, 10))
plot_tree(rf_clf.estimators_[0], feature_names=X_train_l.columns, class_names=rf_clf.classes_, filled=True)
plt.title('Visualization of One Tree in Random Forest')
plt.show()
