# NLP Payments Lab: FinTech Text Classification & Escalation Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end NLP engineering framework designed to classify, interpret, and route payment complaints in modern financial technology pipelines.

Try the live demo: https://huggingface.co/spaces/mitalidaduria/payment-fraud-detector

This repository demonstrates the progressive evolution of domain-specific NLP systems—starting from low-latency, interpretable baselines (TF-IDF + Logistic Regression) and scaling toward fine-tuned Transformer models (DistilBERT), Retrieval-Augmented Generation (RAG), and autonomous LLM customer operations agents.

---

## Executive Overview

In enterprise payment systems, incoming customer communications (chargebacks, failed processing, authorization issues, 2FA locks) must be triaged accurately and instantly. Misclassifying a fraudulent transaction request as a routine refund query leads to regulatory compliance penalties and poor customer retention.

While modern LLMs excel at general language understanding, production FinTech systems often require:
1. **Microsecond Latency:** Real-time routing before hit by heavy LLM inference costs.
2. **Deterministic Interpretability:** Compliance officers must understand *why* a complaint was categorized under fraud or disputed billing.
3. **High Performance on Sparse Data:** Maintaining robust signal strength when labeled samples are limited.

This repository establishes a benchmark system addressing these exact constraints.

---

## Project Architecture & System Evolution

The project follows an iterative engineering progression:
* **Phase 1 (Current):** Bigram TF-IDF vectorization with sublinear term-frequency scaling and balanced Logistic Regression.
* **Phase 2 (Upcoming):** Transfer learning via fine-tuned Transformer architectures (`distilbert-base-uncased`) to capture complex semantic context.
* **Phase 3 (Upcoming):** Vector search and RAG integration for dynamic policy lookup and autonomous escalation routing.

---

##  Payment Complaint Taxonomy

The classifier categorizes incoming text into four core operational categories matching enterprise payment routing workflows:

| Category | Description | Key Triggers & Signals |
| :--- | :--- | :--- |
| `payment_failure` | Technical processing errors, decline codes, or gateway issues | *declined, gateway error, code 504, timeout* |
| `fraud_dispute` | Unauthorized account activity or suspected fraudulent charges | *unauthorized, unrecognized charge, stolen card* |
| `refund_request` | Post-transaction disputes, double charges, or pending refunds | *charged twice, double charge, missing refund* |
| `account_issue` | Authentication errors, account locks, and balance discrepancies | *2FA failure, account locked, balance discrepancy* |

---

##  Repository Structure

```text
nlp-payments-lab/
├── src/
│   ├── data/
│   │   └── make_dataset.py        # Synthetic payment complaint data generator
│   └── models/
│       └── tfidf_classifier.py    # TF-IDF + Logistic Regression pipeline
├── main.py                        # Training, evaluation, and interpretability entrypoint
├── requirements.txt               # Dependencies
└── README.md                      # System documentation
