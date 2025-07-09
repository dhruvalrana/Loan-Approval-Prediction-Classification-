import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, log_loss
import seaborn as sns

# Load the dataset
df = pd.read_csv(r'Loan_Approval_Prediction.csv') # File path to your dataset

# Data Preprocessing
X_loan = df.drop(['loan_id', 'loan_status'], axis=1)
y_loan = df['loan_status']
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(X_loan, y_loan, test_size=0.2, random_state=42)

# Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Encode categorical variables
label_encoder = LabelEncoder()
for column in X_train_l.select_dtypes(include=['object']).columns:
    X_train_l[column] = label_encoder.fit_transform(X_train_l[column])
    X_test_l[column] = label_encoder.transform(X_test_l[column])

# Fit the model
clf.fit(X_train_l, y_train_l)

# Predictions
y_pred_l = clf.predict(X_test_l)

# Evaluation Metrics
accuracy_l = accuracy_score(y_test_l, y_pred_l)
precision_l = precision_score(y_test_l, y_pred_l, pos_label='Approved')
recall_l = recall_score(y_test_l, y_pred_l, pos_label='Approved')
f1_l = f1_score(y_test_l, y_pred_l, pos_label='Approved')
log_loss_l = log_loss(y_test_l, clf.predict_proba(X_test_l))

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

# Decision Tree for House Price Prediction (Regression)
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.tree import DecisionTreeClassifier, plot_tree
# # Load the dataset
# df = pd.read_csv(r'D:\BIA\Sem - 3\M_L_Lokesh Sir\House Price Prediction (Regression)\House_Price_Prediction.csv')

# # Prepare the data
# X = df[['house_id', 'area_sqft', 'bedrooms', 'bathrooms', 'location_rating', 'age_years']]
# y = df['price']

# # Encode categorical variables if necessary
# label_encoder = LabelEncoder()

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create and fit the model
# model = DecisionTreeClassifier(random_state=42)
# model.fit(X_train, y_train)

# # Make predictions
# y_pred = model.predict(X_test)

# # Calculate the accuracy
# accuracy = model.score(X_test, y_test)
# print(f'Accuracy: {accuracy}')

# # Visualize the decision tree
# plt.figure(figsize=(12, 8))
# plot_tree(model, feature_names=X.columns, class_names=np.unique(y).astype(str), filled=True)
# plt.title('Decision Tree for House Price Prediction')
# plt.show()
