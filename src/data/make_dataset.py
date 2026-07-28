import random
import pandas as pd

TEMPLATES = {
    "payment_failure": [
        "Payment failed due to a gateway timeout when checking out.",
        "My credit card was declined at checkout with error code 504.",
        "Transaction timed out while processing payment on the portal.",
        "Card payment failed twice during final payment step.",
        "Gateway error occurred while trying to pay for order."
    ],
    "fraud_dispute": [
        "Unexplained charge on my statement from an unrecognized merchant.",
        "Unauthorized transaction detected on my account last night.",
        "I need to dispute a fraudulent transaction made without my consent.",
        "Someone used my card details illegally for an online purchase.",
        "Disputing chargeback for transaction I did not authorize."
    ],
    "refund_request": [
        "I was charged twice for a single order, please refund the extra.",
        "Cancelled my order yesterday but haven't received my refund.",
        "Requesting a full refund due to item never arriving.",
        "Double charge appeared on my invoice for order #1234.",
        "Money deducted from bank but order shows pending refund."
    ],
    "account_issue": [
        "Unable to log into my account due to 2FA verification failure.",
        "My account is locked after entering password incorrectly.",
        "Balance showing incorrectly after my last deposit.",
        "Two-factor authentication code is not sending to my phone.",
        "Account suspended for suspicious activity, please help unlock."
    ]
}

def generate_synthetic_complaints(num_samples=1200, random_state=42):
    random.seed(random_state)
    data = []
    categories = list(TEMPLATES.keys())
    
    for _ in range(num_samples):
        cat = random.choice(categories)
        phrase = random.choice(TEMPLATES[cat])
        prefix = random.choice(["Hi, ", "Help, ", "Urgent: ", "", "Please resolve: "])
        suffix = random.choice([" Please fix.", " Thank you.", " Need urgent assistance.", ""])
        text = f"{prefix}{phrase}{suffix}"
        data.append({"text": text, "category": cat})
        
    return pd.DataFrame(data)

if __name__ == "__main__":
    df = generate_synthetic_complaints(1200)
    df.to_csv("src/data/complaints.csv", index=False)
    print("Successfully generated 1,200 synthetic payment complaints at src/data/complaints.csv")
