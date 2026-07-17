import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("dataset/student_placement.csv")

# Remove salary column because we are predicting placement status
df = df.drop("salary_package_lpa", axis=1)

# Convert categorical columns into numerical values
le_branch = LabelEncoder()
df["branch"] = le_branch.fit_transform(df["branch"])

le_tier = LabelEncoder()
df["college_tier"] = le_tier.fit_transform(df["college_tier"])

# Separate features (input) and target (output)
X = df.drop("placement_status", axis=1)
y = df["placement_status"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create the machine learning model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Feature Importance

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(10,5))

sns.barplot(
    data=importance,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance in Placement Prediction")
plt.xlabel("Importance")
plt.ylabel("Features")

plt.savefig("model/feature_importance.png")
plt.close()

# Make predictions
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", accuracy)

print(classification_report(y_test, y_pred))

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(5,4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.savefig("model/confusion_matrix.png")
plt.close()

# Save the trained model
joblib.dump(model, "model/placement_model.pkl")

# Save the LabelEncoders
joblib.dump(le_branch, "model/branch_encoder.pkl")
joblib.dump(le_tier, "model/tier_encoder.pkl")

print("Model and encoders saved successfully!")