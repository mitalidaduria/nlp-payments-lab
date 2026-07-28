from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from src.data.make_dataset import generate_synthetic_complaints
from src.models.tfidf_classifier import PaymentComplaintClassifier

def run_pipeline():
    print("1. Generating synthetic payment complaints dataset...")
    df = generate_synthetic_complaints(num_samples=1200)
    
    X = df['text']
    y = df['category']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("2. Training TF-IDF + Logistic Regression Baseline...")
    model = PaymentComplaintClassifier()
    model.fit(X_train, y_train)
    
    print("\n3. Classification Metrics:")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    print("\n4. Top Driving Words Per Category (Interpretability):")
    model.top_features_per_class(n=5)

if __name__ == "__main__":
    run_pipeline()
