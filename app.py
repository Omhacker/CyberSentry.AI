import re
import collections.abc
import collections
collections.Sequence = collections.abc.Sequence

import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from urllib.parse import urlparse
from datetime import datetime
# import whois  # optional, comment for deployment simplicity

# --- 1. Load & Preprocess Dataset ---
df = pd.read_csv("CEAS_08.csv.gz")  # Make sure this file exists in your root

df['subject'] = df['subject'].fillna("No Subject")
df['receiver'] = df['receiver'].fillna("Unknown Receiver")
df['email_text'] = df['subject'] + ' ' + df['body']

def clean_email(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"<.*?>", '', text)
    text = re.sub(r"[^\w\s]", '', text)
    return text.lower().strip()

df['clean_email'] = df['email_text'].apply(clean_email)

# --- 2. TF-IDF Vectorizer & Model Training ---
X = df['clean_email']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_df=0.85,
    min_df=5,
    ngram_range=(1, 2),
    max_features=5000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# --- 3. Flask App Setup ---
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# --- 4. Serve Frontend HTML ---
@app.route('/')
def serve_index():
    return send_from_directory('static', 'index.html')


# --- 5. API: Email Phishing Detection ---
@app.route('/predict-email', methods=['POST'])
def predict_email():
    data = request.json
    text = data.get("text", "")
    cleaned = clean_email(text)
    vec = vectorizer.transform([cleaned])
    pred_prob = model.predict_proba(vec)[0][1]
    pred = int(pred_prob > 0.4)  # Threshold
    return jsonify({
        "prediction": pred,
        "confidence": round(pred_prob * 100, 2)
    })


# --- 6. Link Extraction Utility ---
def extract_links(text):
    return re.findall(r'https?://\S+', text)


# --- 7. Scoring Function for Link Threats ---
def score_link(link):
    score = 0
    reasons = []
    parsed = urlparse(link)
    domain = parsed.netloc

    if re.search(r'https?://\d+\.\d+\.\d+\.\d+', link):
        score += 3
        reasons.append("Uses IP address")

    if any(domain.endswith(tld) for tld in ['.ru', '.xyz', '.tk', '.top', '.ml']):
        score += 2
        reasons.append("Suspicious TLD")

    common_spoofs = ['paypal', 'amazon', 'google', 'facebook', 'netflix']
    for brand in common_spoofs:
        if brand in domain and not re.match(rf'^{brand}\.com$', domain):
            score += 3
            reasons.append(f"Spoofing: {brand}")

    spoofed_lookalikes = ['paypaI', 'arnazon', 'goog1e', 'rnicrosoft', 'facebo0k']
    if any(look in domain for look in spoofed_lookalikes):
        score += 4
        reasons.append("Spoofed domain (lookalike characters)")

    if domain.count('.') > 3:
        score += 1
        reasons.append("Too many subdomains")

    if '-' in domain:
        score += 1
        reasons.append("Hyphens in domain")

    if any(word in link.lower() for word in ['login', 'verify', 'secure', 'update', 'bank', 'free']):
        score += 2
        reasons.append("Suspicious keywords")

    if parsed.scheme != "https":
        score += 1
        reasons.append("Not using HTTPS")

    risk = 'High' if score >= 6 else 'Medium' if score >= 3 else 'Low'

    return {
        'link': link,
        'score': score,
        'risk': risk,
        'reasons': reasons,
        'domain_age_days': "❓ Unknown"  # Placeholder, remove if using whois
    }


# --- 8. API: Link Analyzer ---
@app.route('/analyze-links', methods=['POST'])
def analyze_links():
    data = request.json
    text = data.get("text", "")
    links = extract_links(text)
    results = []
    for link in links:
        result = score_link(link)
        results.append(result)
    return jsonify(results)


# --- 9. Run Flask App ---
if __name__ == '__main__':
    app.run(debug=True)
