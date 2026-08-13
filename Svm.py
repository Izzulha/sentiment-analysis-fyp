import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- CONFIGURATION ---
TARGET_COLUMN_CANDIDATE = 'score'

STRICT_PARAMS = {
    'C': [100],                 # powerful classifier
    'gamma': [0.001],           # smooth but sensitive
    'kernel': ['rbf'],
    'class_weight': ['balanced']
}

# ---------------------------------------------------------

def train_strict_svm():
    # 1. Load Data
    file_name = 'Svm and rf modelling data 2.csv'
    try:
        df = pd.read_csv(file_name, sep=',')
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    df.columns = df.columns.str.strip()

    # ---------------------------------------------------------
    #     TARGET COLUMN DETECTION
    # ---------------------------------------------------------
    normalized_cols_map = {col.lower(): col for col in df.columns}
    detected_target_name = None

    # 1. Exact match
    if TARGET_COLUMN_CANDIDATE.lower() in normalized_cols_map:
        detected_target_name = normalized_cols_map[TARGET_COLUMN_CANDIDATE.lower()]
    else:
        # 2. Partial match
        for lower_col, original_col in normalized_cols_map.items():
            if TARGET_COLUMN_CANDIDATE.lower() in lower_col:
                detected_target_name = original_col
                break

    if detected_target_name is None:
        print("="*50)
        print(f"CRITICAL ERROR: Target column containing '{TARGET_COLUMN_CANDIDATE}' not found.")
        print(f"Available Columns: {df.columns.tolist()}")
        print("="*50)
        return

    TARGET_COLUMN = detected_target_name
    print(f"Successfully detected target column: '{TARGET_COLUMN}'")

    # ---------------------------------------------------------
    #     DATA CLEANING
    # ---------------------------------------------------------
    initial_shape = df.shape
    df = df.dropna()
    final_shape = df.shape

    print("\n--- Data Cleaning Summary ---")
    print(f"Original rows : {initial_shape[0]}")
    print(f"Rows dropped  : {initial_shape[0] - final_shape[0]}")
    print(f"Final rows    : {final_shape[0]}")

    if final_shape[0] == 0:
        print("ERROR: Dataset empty after cleaning.")
        return

    # Prepare X and y
    y = df[TARGET_COLUMN]
    X = df.drop(TARGET_COLUMN, axis=1)

    # Convert features to numeric
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Drop rows with NaN after conversion
    X = X.dropna()
    y = y.loc[X.index]

    print(f"Rows remaining after numeric cleaning: {len(X)}")

    if len(y.unique()) < 2:
        print("ERROR: Only one class detected.")
        return
    if len(X) < 10:
        print("ERROR: Not enough samples.")
        return

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.4, random_state=42, stratify=y_encoded
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------------------------------------------
    #     STRICT SVM TRAINING (Corrected parameters)
    # ---------------------------------------------------------
    print("\n" + "="*60)
    print(f"TRAINING WITH STRICT PARAMETERS:")
    print(STRICT_PARAMS)
    print("="*60)

    best_model = SVC(
        C=STRICT_PARAMS['C'][0],
        gamma=STRICT_PARAMS['gamma'][0],
        kernel=STRICT_PARAMS['kernel'][0],
        class_weight=STRICT_PARAMS['class_weight'][0],
        random_state=42
    )

    best_model.fit(X_train_scaled, y_train)

    # ---------------------------------------------------------
    #     EVALUATION
    # ---------------------------------------------------------
    y_pred = best_model.predict(X_test_scaled)

    y_test_decoded = le.inverse_transform(y_test)
    y_pred_decoded = le.inverse_transform(y_pred)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nFinal Strict SVM Test Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded, zero_division=0))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_decoded, y_pred_decoded))


if __name__ == "__main__":
    train_strict_svm()
