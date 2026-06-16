import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# =========================
# 1. LOAD FUNCTION
# =========================

def load_uci_file(filepath):
    column_names = [
        "age", "sex", "cp", "trestbps", "chol", "fbs",
        "restecg", "thalach", "exang", "oldpeak",
        "slope", "ca", "thal", "target"
    ]

    df = pd.read_csv(
        filepath,
        header=None,
        names=column_names,
        na_values="?",
        sep=","
    )

    df = df.apply(pd.to_numeric, errors='coerce')
    return df


# =========================
# 2. LOAD DATASETS
# =========================

df1 = load_uci_file("processed.cleveland.data")
df2 = load_uci_file("processed.hungarian.data")
df3 = load_uci_file("processed.switzerland.data")
df4 = load_uci_file("processed.va.data")

# Merge all
df = pd.concat([df1, df2, df3, df4], ignore_index=True)

print("Dataset loaded:", df.shape)

# =========================
# 3. CLEAN DATA
# =========================

# Handle missing values
imputer = SimpleImputer(strategy="median")
df.iloc[:, :-1] = imputer.fit_transform(df.iloc[:, :-1])

# Convert target to binary
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

print("After cleaning:", df.shape)

# =========================
# 4. SPLIT DATA
# =========================

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =========================
# 5. TRAIN MODEL
# =========================

model = XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05
)

model.fit(X_train, y_train)

print("✅ Model trained successfully")


# =========================
# 6. PREDICTION FUNCTION
# =========================

def predict(input_data, threshold=0.6):
    """
    input_data: list of 13 features
    returns: prediction (0/1), probability
    """

    input_array = np.array(input_data).reshape(1, -1)

    probability = model.predict_proba(input_array)[0][1]
    prediction = 1 if probability > threshold else 0

    return prediction, probability