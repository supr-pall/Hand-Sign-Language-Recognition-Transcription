import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join("data", "hand_signs.csv")
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "hand_sign_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

def main():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Dataset not found at {DATA_PATH}. Run collect_data.py first.")
        return

    df = pd.read_csv(DATA_PATH)
    if "label" not in df.columns:
        print("[ERROR] 'label' column not found in dataset.")
        return

    X = df.drop("label", axis=1).values
    y = df["label"].values

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    clf.fit(X_train, y_train)

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    print(f"[INFO] Training Accuracy: {train_acc:.3f}")
    print(f"[INFO] Test Accuracy: {test_acc:.3f}")

    model_bundle = {
        "classifier": clf,
        "label_encoder": le
    }

    joblib.dump(model_bundle, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
