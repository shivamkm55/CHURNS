# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE # PPT Feature: Handle Imbalance
from sklearn.metrics import accuracy_score

# 1. LOAD DATA
# Ensure 'churn.csv' is in your folder
df = pd.read_csv('churn.csv')

# 2. DATA CLEANING
# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
# Drop ID
if 'customerID' in df.columns:
    df = df.drop('customerID', axis=1)

# 3. PPT FEATURE: BEHAVIORAL TRACKING SIMULATION
# Since we don't have real click data, we simulate it for the demo
np.random.seed(42)
df['Unanswered_Emails'] = np.random.randint(0, 5, size=len(df))
df['Support_Ticket_Clicks'] = np.random.randint(0, 10, size=len(df))

# 4. ENCODING
# Encode Target
le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])
# Encode Features
df_encoded = pd.get_dummies(df, drop_first=True)

# 5. HANDLE IMBALANCE (SMOTE)
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

print("Applying SMOTE to balance data...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 6. TRAIN & SELECT BEST MODEL (PPT Feature: Model Suite)
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

best_model = None
best_acc = 0

print("Training models...")
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f" -> {name} Accuracy: {acc:.4f}")
    
    if acc > best_acc:
        best_acc = acc
        best_model = model

# 7. SAVE ARTIFACTS
# We need to save the columns list to ensure input order matches later
joblib.dump(best_model, 'best_churn_model.pkl')
joblib.dump(X.columns.tolist(), 'model_columns.pkl')
print(f"✅ Success! Saved {best_model.__class__.__name__} as 'best_churn_model.pkl'")