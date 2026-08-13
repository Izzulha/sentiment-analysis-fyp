import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder


def load_and_fix_data(filepath):
    separators = [',', ';', '\t']

    for sep in separators:
        try:
            df = pd.read_csv(filepath, sep=sep)

            # If only one column, separator is wrong
            if df.shape[1] < 2:
                continue

            df.columns = df.columns.str.strip()

            # Find target column
            target_col = None
            for col in df.columns:
                if "score" in col.lower():
                    target_col = col
                    break

            if target_col:
                print(f"-> Success using separator '{sep}'")
                print(f"-> Found target column: '{target_col}'")
                df.rename(columns={target_col: 'Score'}, inplace=True)
                return df

        except Exception:
            continue

    return None


def train_rf_final():
    print("\n--- 1. Loading Data ---")
    df = load_and_fix_data('Svm and rf modelling data 2.csv')

    if df is None:
        print("CRITICAL ERROR: Could not read the file.")
        return

    # --- 2. Data Cleaning ---
    y = df['Score']
    X = df.drop('Score', axis=1)

    print(f"Features detected: {X.shape[1]}")

    X = X.dropna(axis=1, how='all')

    print("Converting features to numeric...")
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')

    data_clean = pd.concat([X, y], axis=1).dropna()
    print(f"Rows used for training: {len(data_clean)}")

    if len(data_clean) < 10:
        print("ERROR: Too many rows dropped.")
        return

    X_clean = data_clean.drop('Score', axis=1)
    y_clean = data_clean['Score']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_clean)

    class_counts = pd.Series(y_encoded).value_counts()
    stratify_param = y_encoded if class_counts.min() >= 2 else None

    # --- 3. Train-test split ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=stratify_param
    )

    # --- 4. Random Forest Model (YOUR PARAMETERS) ---
    rf_model = RandomForestClassifier(
        n_estimators=80,
        max_depth=8,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        max_features=0.3
    )

    print("\n--- 5. Training Random Forest Model ---")
    rf_model.fit(X_train, y_train)

    # --- 6. Evaluation ---
    y_pred = rf_model.predict(X_test)

    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nFinal Accuracy: {accuracy * 100:.2f}%")

    print("\n--- Classification Report ---")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

    # --- 7. Confusion Matrix ---
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)

    print("\n--- Confusion Matrix (Table) ---")
    print(cm_df)

    # --- 8. Confusion Matrix Plot (IMAGE) ---
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_df,
        annot=True,
        fmt='d',
        cmap='Blues',
        cbar=True
    )

    plt.title("Confusion Matrix for Random Forest Model")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()

    # Save high-quality image for thesis
    plt.savefig("confusion_matrix_random_forest.png", dpi=300)
    plt.show()

    # --- 9. Feature Importance ---
    print("\n--- Top 10 Most Important Features ---")
    feature_imp_df = pd.DataFrame({
        'Feature': X_clean.columns,
        'Importance': rf_model.feature_importances_
    })

    print(
        feature_imp_df
        .sort_values(by='Importance', ascending=False)
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    train_rf_final()
