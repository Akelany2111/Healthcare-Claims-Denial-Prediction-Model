from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load data
claims = pd.read_csv("claims.csv")
providers = pd.read_csv("providers.csv")
insurance = pd.read_csv("insurance.csv")

# Merge tables
df = claims.merge(providers, on="provider_id", how="left")
df = df.merge(insurance, on="insurance_id", how="left")


# Preview
print(df.head())
print(df.shape)
print(df.columns)


print(df["denial_flag"].value_counts())
print(df["denial_flag"].value_counts(normalize=True))

# Convert date columns
df["service_date"] = pd.to_datetime(df["service_date"])
df["submission_date"] = pd.to_datetime(df["submission_date"])

# Date-based features
df["service_year"] = df["service_date"].dt.year
df["service_month"] = df["service_date"].dt.month
df["service_dayofweek"] = df["service_date"].dt.dayofweek

# Days between service and submission
df["days_to_submit"] = (df["submission_date"] - df["service_date"]).dt.days

# Preview
print(df[["service_date", "submission_date", "service_year",
      "service_month", "service_dayofweek", "days_to_submit"]].head())

# Select features and target
feature_cols = [
    "provider_name",
    "specialty",
    "clinic_location",
    "payer_name",
    "payer_type",
    "claim_profile",
    "billed_amount",
    "service_year",
    "service_month",
    "service_dayofweek",
    "days_to_submit"
]

target_col = "denial_flag"

X = df[feature_cols]
y = df[target_col]

categorical_features = [
    "provider_name",
    "specialty",
    "clinic_location",
    "payer_name",
    "payer_type",
    "claim_profile"
]

numeric_features = [
    "billed_amount",
    "service_year",
    "service_month",
    "service_dayofweek",
    "days_to_submit"
]

# Build preprocessing pipeline

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Preprocessing
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", categorical_transformer, categorical_features),
        ("num", numeric_transformer, numeric_features)
    ]
)

# Train the first model

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(X_train, y_train)

# make predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# evaluate the model

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nROC AUC Score:", roc_auc_score(y_test, y_prob))

# try a stronger model
# now we train a Random Forest, which often performs better.

rf_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ))
])

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
rf_prob = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))
print("\nRandom Forest Classification Report:\n",
      classification_report(y_test, rf_pred))
print("\nRandom Forest Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("\nRandom Forest ROC AUC Score:", roc_auc_score(y_test, rf_prob))

# Feature importance. This helps explain what drives denial risk.

# Get transformed feature names
ohe = rf_model.named_steps["preprocessor"].named_transformers_[
    "cat"].named_steps["onehot"]
cat_names = ohe.get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([cat_names, numeric_features])

# Get importances
importances = rf_model.named_steps["classifier"].feature_importances_

feature_importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print(feature_importance_df.head(15))


# Plot top features

top_features = feature_importance_df.head(10)

plt.figure(figsize=(10, 6))
plt.barh(top_features["feature"], top_features["importance"])
plt.gca().invert_yaxis()
plt.title("Top 10 Features Driving Claim Denial Prediction")
plt.xlabel("Importance")
plt.tight_layout()
plt.show()
class_weight = "balanced"

ConfusionMatrixDisplay.from_predictions(y_test, rf_pred)
plt.title("Confusion Matrix - Claim Denial Prediction")
plt.show()
