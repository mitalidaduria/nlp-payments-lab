import os
from typing import List, Dict, Union
from transformers import pipeline


class TransformerComplaintClassifier:
    """Zero-shot complaint classifier using HuggingFace BART-large-mnli running on CPU."""

    def __init__(self, model_name: str = "facebook/bart-large-mnli"):
        self.model_name = model_name
        self.classifier = None

    def load(self) -> None:
        """Loads the pre-trained zero-shot classification pipeline on CPU."""
        print(f"Loading zero-shot pipeline: {self.model_name}...")
        # device=-1 forces CPU execution
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=-1
        )
        print("Model loaded successfully!")

    def predict(
        self, 
        texts: Union[str, List[str]], 
        candidate_labels: List[str]
    ) -> List[Dict]:
        """Classifies text into candidate categories without fine-tuning."""
        if self.classifier is None:
            raise RuntimeError("Model is not loaded. Call .load() first.")
        
        if isinstance(texts, str):
            texts = [texts]

        results = self.classifier(texts, candidate_labels=candidate_labels)
        return results if isinstance(results, list) else [results]

    def compare_with_tfidf(
        self, 
        text: str, 
        candidate_labels: List[str], 
        keywords_dict: Dict[str, List[str]]
    ) -> Dict:
        """Demonstrates keyword matching vs zero-shot semantic inference."""
        # 1. Direct Keyword / TF-IDF Lookup
        text_lower = text.lower()
        tfidf_predicted = "unknown"
        for label, keywords in keywords_dict.items():
            if any(kw in text_lower for kw in keywords):
                tfidf_predicted = label
                break

        # 2. Zero-Shot Semantic Classification
        transformer_res = self.predict(text, candidate_labels)[0]
        top_label = transformer_res["labels"][0]
        top_score = transformer_res["scores"][0]

        return {
            "text": text,
            "tfidf_keyword_match": tfidf_predicted,
            "transformer_prediction": top_label,
            "confidence_score": round(top_score, 4)
        }


if __name__ == "__main__":
    clf = TransformerComplaintClassifier()
    clf.load()

    labels = ["refund_request", "unauthorized_charge", "delivery_issue", "account_access"]
    keywords = {
        "refund_request": ["refund", "reimburse", "money back"],
        "unauthorized_charge": ["unauthorized", "fraud", "stolen"],
        "delivery_issue": ["delivery", "delayed", "shipment"]
    }

    sample = "Card was charged but item never arrived"
    res = clf.compare_with_tfidf(sample, labels, keywords)
    print("\n--- Test Comparison ---")
    print(f"Text: '{res['text']}'")
    print(f"TF-IDF Match: {res['tfidf_keyword_match']}")
    print(f"Transformer Prediction: {res['transformer_prediction']} (Score: {res['confidence_score']})")
