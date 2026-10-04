import os
import re
import string
import csv
from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = Flask(__name__)

# Standard English stopwords set (no external nltk download required)
STOPWORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 
    'aren\'t', 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 
    'but', 'by', 'can', 'can\'t', 'cannot', 'could', 'couldn\'t', 'did', 'didn\'t', 'do', 'does', 
    'doesn\'t', 'doing', 'don\'t', 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', 
    'hadn\'t', 'has', 'hasn\'t', 'have', 'haven\'t', 'having', 'he', 'he\'d', 'he\'ll', 'he\'s', 
    'her', 'here', 'here\'s', 'hers', 'herself', 'him', 'himself', 'his', 'how', 'how\'s', 'i', 
    'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'in', 'into', 'is', 'isn\'t', 'it', 'it\'s', 'its', 
    'itself', 'let\'s', 'me', 'more', 'most', 'mustn\'t', 'my', 'myself', 'no', 'nor', 'not', 
    'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 
    'out', 'over', 'own', 'same', 'shan\'t', 'she', 'she\'d', 'she\'ll', 'she\'s', 'should', 
    'shouldn\'t', 'so', 'some', 'such', 'than', 'that', 'that\'s', 'the', 'their', 'theirs', 
    'them', 'themselves', 'then', 'there', 'there\'s', 'these', 'they', 'they\'d', 'they\'ll', 
    'they\'re', 'they\'ve', 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 
    'very', 'was', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'re', 'we\'ve', 'were', 'weren\'t', 
    'what', 'what\'s', 'when', 'when\'s', 'where', 'where\'s', 'which', 'while', 'who', 'who\'s', 
    'whom', 'why', 'why\'s', 'with', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll', 
    'you\'re', 'you\'ve', 'your', 'yours', 'yourself', 'yourselves'
}

def preprocess_text(text, do_lower=True, remove_punct=True, remove_stop=True):
    """
    NLP Preprocessing Pipeline:
    1. Lowercasing
    2. Punctuation removal
    3. Stopword removal
    4. Token cleanup
    """
    if not isinstance(text, str):
        return ""
    
    # 1. Lowercase
    if do_lower:
        text = text.lower()
        
    # 2. Punctuation Removal
    if remove_punct:
        text = re.sub(r'[^\w\s]', ' ', text)
        
    # Split into tokens
    tokens = text.split()
    
    # 3. Stopword Removal
    if remove_stop:
        tokens = [token for token in tokens if token.lower() not in STOPWORDS and not token.isdigit()]
        
    cleaned_text = " ".join(tokens)
    return cleaned_text, tokens

# Global model state
vectorizer = None
model = None
dataset_stats = {}
training_data = []

def load_and_train():
    global vectorizer, model, dataset_stats, training_data
    
    dataset_path = os.path.join(os.path.dirname(__file__), 'dataset.csv')
    messages = []
    labels = []
    
    if os.path.exists(dataset_path):
        with open(dataset_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                lbl = row['label'].strip().lower()
                msg = row['message'].strip()
                labels.append(lbl)
                messages.append(msg)
                training_data.append({"label": lbl, "message": msg})
    else:
        raise FileNotFoundError("dataset.csv not found!")

    # Preprocess all training messages
    cleaned_messages = [preprocess_text(msg)[0] for msg in messages]

    # Split dataset for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        cleaned_messages, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # Initialize TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train Multinomial Naive Bayes Model
    model = MultinomialNB(alpha=1.0)
    model.fit(X_train_vec, y_train)

    # Evaluate performance
    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label='spam', zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label='spam', zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label='spam', zero_division=0)

    # Train on full dataset for live predictions
    X_full_vec = vectorizer.fit_transform(cleaned_messages)
    model.fit(X_full_vec, labels)

    spam_count = sum(1 for l in labels if l == 'spam')
    ham_count = sum(1 for l in labels if l == 'ham')
    vocab_size = len(vectorizer.get_feature_names_out())

    dataset_stats = {
        "total_samples": len(labels),
        "spam_samples": spam_count,
        "ham_samples": ham_count,
        "vocabulary_size": vocab_size,
        "accuracy": round(acc * 100, 2),
        "precision": round(prec * 100, 2),
        "recall": round(rec * 100, 2),
        "f1_score": round(f1 * 100, 2)
    }
    
    print(f"Model trained successfully. Accuracy: {dataset_stats['accuracy']}% | Vocab Size: {vocab_size}")

# Initialize model at start
load_and_train()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify(dataset_stats)

@app.route('/api/examples', methods=['GET'])
def get_examples():
    spam_examples = [
        {"id": 1, "title": "Prize Winner Scam", "text": "WINNER! You have won a $1,000 Walmart Gift Card! Click here to claim your prize now http://bit.ly/claim-prize"},
        {"id": 2, "title": "Bank Security Phishing", "text": "URGENT! Your bank account has been locked due to suspicious activity. Verify now at http://secure-bank-alert.com"},
        {"id": 3, "title": "Free iPhone Promo", "text": "CONGRATULATIONS! You have been selected for a free iPhone 15 Pro. Call 1-800-555-0199 immediately to claim!"},
        {"id": 4, "title": "Package Delivery Alert", "text": "ALERT: Your package delivery failed. Update your delivery address now at http://express-logistics-track.info"},
        {"id": 5, "title": "Instant Loan Offer", "text": "Get instant low rate loans up to $50,000 with 0% interest. Apply now at http://fast-cash-loans.net"}
    ]
    
    ham_examples = [
        {"id": 1, "title": "Lunch Plan", "text": "Hey, are we still meeting for lunch today at 12:30?"},
        {"id": 2, "title": "Grocery Reminder", "text": "Don't forget to buy milk and bread on your way home."},
        {"id": 3, "title": "Lecture Notes", "text": "Can you send me the notes from yesterday's NLP lecture?"},
        {"id": 4, "title": "OTP Verification Code", "text": "Your verification code is 584920. Do not share this code with anyone."},
        {"id": 5, "title": "Meeting Update", "text": "Meeting rescheduled to 3 PM in Conference Room B."}
    ]
    
    return jsonify({
        "spam": spam_examples,
        "ham": ham_examples
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    message = data.get('message', '').strip()
    
    # Preprocessing options from request
    do_lower = data.get('do_lower', True)
    remove_punct = data.get('remove_punct', True)
    remove_stop = data.get('remove_stop', True)

    if not message:
        return jsonify({"error": "Please enter a message to analyze."}), 400

    # 1. Preprocess
    cleaned_text, tokens = preprocess_text(
        message, 
        do_lower=do_lower, 
        remove_punct=remove_punct, 
        remove_stop=remove_stop
    )

    if not cleaned_text:
        # If all words were stripped out
        cleaned_text = message.lower() if do_lower else message

    # 2. Vectorize
    vec = vectorizer.transform([cleaned_text])
    
    # 3. Predict & Probabilities
    probs = model.predict_proba(vec)[0]
    classes = list(model.classes_)
    
    # map classes to ham/spam index
    spam_idx = classes.index('spam') if 'spam' in classes else 1
    ham_idx = classes.index('ham') if 'ham' in classes else 0
    
    spam_prob = float(probs[spam_idx])
    ham_prob = float(probs[ham_idx])
    
    is_spam = spam_prob >= 0.5
    prediction_label = "Spam" if is_spam else "Not Spam"
    confidence = round((spam_prob if is_spam else ham_prob) * 100, 2)

    # Calculate word feature contributions for NLP explanation
    feature_names = vectorizer.get_feature_names_out()
    vec_array = vec.toarray()[0]
    
    # Top informative words present in the message
    word_weights = []
    for idx in vec_array.nonzero()[0]:
        word = feature_names[idx]
        weight = float(vec_array[idx])
        # Find log ratio of spam vs ham for this feature
        spam_log_prob = float(model.feature_log_prob_[spam_idx][idx])
        ham_log_prob = float(model.feature_log_prob_[ham_idx][idx])
        bias = "spam" if spam_log_prob > ham_log_prob else "ham"
        
        word_weights.append({
            "word": word,
            "tfidf": round(weight, 3),
            "bias": bias
        })

    # Sort word weights by TFIDF weight descending
    word_weights = sorted(word_weights, key=lambda x: x["tfidf"], reverse=True)[:8]

    return jsonify({
        "original_text": message,
        "preprocessed_text": cleaned_text,
        "tokens": tokens,
        "prediction": prediction_label,
        "is_spam": is_spam,
        "confidence": confidence,
        "spam_probability": round(spam_prob * 100, 2),
        "ham_probability": round(ham_prob * 100, 2),
        "top_keywords": word_weights,
        "preprocessing_steps": {
            "lowercase": do_lower,
            "punctuation_removed": remove_punct,
            "stopwords_removed": remove_stop
        }
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))
    app.run(host='0.0.0.0', port=port, debug=True)
