import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Necessary imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt

# Load the dataset
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Inspect the data
print("Training Data Sample:")
print(train_data.head())
print("\nTraining Data Info:")
print(train_data.info())

# Ensure 'Exited' column exists in the train_data
if 'Exited' not in train_data.columns:
    raise ValueError("The target column 'Exited' is missing in the training dataset.")

# Get numeric columns common to both train_data and test_data
numeric_cols = train_data.select_dtypes(include=['number']).columns.intersection(test_data.columns)

# Fill numeric columns with their mean
train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].mean())
test_data[numeric_cols] = test_data[numeric_cols].fillna(train_data[numeric_cols].mean())

# Get non-numeric columns common to both train_data and test_data
non_numeric_cols = train_data.select_dtypes(exclude=['number']).columns.intersection(test_data.columns)

# Fill non-numeric columns with their mode
for col in non_numeric_cols:
    mode_value = train_data[col].mode().iloc[0]
    train_data[col] = train_data[col].fillna(mode_value)
    test_data[col] = test_data[col].fillna(mode_value)

# Align test_data columns with train_data
test_data = test_data.reindex(columns=train_data.columns, fill_value=0)

# Separate features and target
X = train_data.drop('Exited', axis=1)
y = train_data['Exited']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle unexpected placeholders ('Nwachukwu') in the data
X_train = X_train.replace('Nwachukwu', np.nan)
X_val = X_val.replace('Nwachukwu', np.nan)

# Fill missing values with 0
X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
X_val = X_val.apply(pd.to_numeric, errors='coerce').fillna(0)

# Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train, y_train)
y_pred_log = log_reg.predict(X_val)

log_reg_roc_auc = roc_auc_score(y_val, log_reg.predict_proba(X_val)[:, 1])
log_reg_f1 = f1_score(y_val, y_pred_log)

print("Logistic Regression - ROC AUC:", log_reg_roc_auc)
print("Logistic Regression - F1 Score:", log_reg_f1)

# Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_val)

dt_roc_auc = roc_auc_score(y_val, dt.predict_proba(X_val)[:, 1])
dt_f1 = f1_score(y_val, y_pred_dt)

print("Decision Tree - ROC AUC:", dt_roc_auc)
print("Decision Tree - F1 Score:", dt_f1)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_val)

rf_roc_auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
rf_f1 = f1_score(y_val, y_pred_rf)

print("Random Forest - ROC AUC:", rf_roc_auc)
print("Random Forest - F1 Score:", rf_f1)

# Model comparison plot
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
roc_scores = [log_reg_roc_auc, dt_roc_auc, rf_roc_auc]
f1_scores = [log_reg_f1, dt_f1, rf_f1]

plt.figure(figsize=(10, 5))
bar_width = 0.35
index = np.arange(len(models))

plt.bar(index, roc_scores, bar_width, alpha=0.6, label='ROC AUC')
plt.bar(index + bar_width, f1_scores, bar_width, alpha=0.6, label='F1 Score')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Comparison')
plt.xticks(index + bar_width / 2, models)
plt.legend()
plt.tight_layout()
plt.show()

# Identify the best model based on ROC AUC scores
best_model = None
if log_reg_roc_auc >= dt_roc_auc and log_reg_roc_auc >= rf_roc_auc:
    best_model = log_reg
    print("Best model: Logistic Regression")
elif dt_roc_auc >= log_reg_roc_auc and dt_roc_auc >= rf_roc_auc:
    best_model = dt
    print("Best model: Decision Tree")
else:
    best_model = rf
    print("Best model: Random Forest")

# Ensure test_data is processed similarly to training data
X_test = test_data.drop('Exited', axis=1, errors='ignore')  # Drop target column if it exists
X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)  # Handle missing or invalid data

# Make predictions with the best model
if hasattr(best_model, "predict_proba"):  # For models that support probabilities
    test_predictions = best_model.predict_proba(X_test)[:, 1]
else:  # For models that don't support probabilities
    test_predictions = best_model.predict(X_test)

# Add the `id` column to predictions
submission = pd.DataFrame({
    'id': test_data['id'],  # Assuming `id` exists in the test dataset
    'Exited': test_predictions
})

# Save predictions to a CSV file
submission.to_csv('submission.csv', index=False)
print("Submission file saved as 'submission.csv'")
