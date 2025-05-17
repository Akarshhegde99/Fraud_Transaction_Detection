import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    # Load dataset
    df = pd.read_csv('creditcard.csv')

    # Prepare data
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, 'fraud_model.pkl')
    print("✅ Model trained and saved as fraud_model.pkl")

# Run the training if this file is executed
if __name__ == "__main__":
    train_and_save_model()
