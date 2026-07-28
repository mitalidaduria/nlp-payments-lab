from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

class PaymentComplaintClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),  # Bigrams: capture 'payment failed' vs 'payment'
                sublinear_tf=True,   # Logarithmic scaling reduces dominance of high-freq words
                stop_words='english'
            )),
            ('clf', LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight='balanced',  # Handles potential class imbalance
                random_state=42
            ))
        ])

    def fit(self, X_train, y_train):
        self.pipeline.fit(X_train, y_train)

    def predict(self, X):
        return self.pipeline.predict(X)

    def top_features_per_class(self, n=5):
        """Prints feature weights to provide compliance-friendly explanations."""
        tfidf = self.pipeline.named_steps['tfidf']
        clf = self.pipeline.named_steps['clf']
        names = tfidf.get_feature_names_out()
        
        for i, cat in enumerate(clf.classes_):
            top_indices = clf.coef_[i].argsort()[-n:]
            top_words = names[top_indices]
            print(f"[{cat}]: {', '.join(top_words[::-1])}")
