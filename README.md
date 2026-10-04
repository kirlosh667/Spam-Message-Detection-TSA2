# SMS Spam Message Detection Web Application (NLP Lab Project)

A machine learning web application for classifying SMS messages as **Spam** or **Not Spam (Legitimate)** using **Natural Language Processing (NLP)** techniques, **TF-IDF Vectorization**, and **Multinomial Naive Bayes Classification**.

---

## 📌 Project Overview
This project provides an interactive NLP environment for:
1. **Entering custom SMS or message text** to evaluate live predictions.
2. **Preprocessing Pipeline Customization** (Toggle lowercasing, punctuation removal, stopword removal).
3. **Probability & Confidence Scoring** ($P(\text{Spam} \mid \text{Text})$ vs $P(\text{Legitimate} \mid \text{Text})$).
4. **Keyword Feature Analysis** (Highlights top TF-IDF features and their class bias).
5. **Interactive Sample Testing** (Preset spam and legitimate messages for instant testing).

---

## 🛠 Tech Stack
- **Backend**: Python 3.x, Flask
- **Machine Learning**: `scikit-learn` (`TfidfVectorizer`, `MultinomialNB`), `numpy`
- **Data Handling**: `csv`, `re`
- **Frontend**: HTML5, Vanilla CSS3 (Custom Dark Cyberpunk/Lab Aesthetic), Vanilla JavaScript (ES6 AJAX)

---

## 🔬 NLP Machine Learning Methodology

### 1. Preprocessing Pipeline
Text preprocessing transforms raw SMS text into clean normalized tokens:
- **Lowercasing**: Normalizes case differences (e.g., `"WINNER"` $\to$ `"winner"`).
- **Punctuation Removal**: Strips special characters, symbols, and URLs (e.g., `"http://..."`, `"!"`, `"$"`).
- **Stopword Removal**: Eliminates uninformative words (e.g., `"the"`, `"is"`, `"at"`, `"you"`).

### 2. Feature Extraction (TF-IDF Vectorization)
The cleaned text is vectorized using **Term Frequency - Inverse Document Frequency (TF-IDF)** with unigram and bigram features:

$$\text{TF-IDF}(t, d, D) = \text{TF}(t, d) \times \text{IDF}(t, D)$$

Where:
- $\text{TF}(t, d)$ is the frequency of word $t$ in document $d$.
- $\text{IDF}(t, D) = \log\left(\frac{N}{|\{d \in D : t \in d\}|}\right)$ adjusts for word commonality across all messages.

### 3. Classification (Multinomial Naive Bayes)
Using Bayes' Theorem under the conditional independence assumption:

$$P(\text{Spam} \mid X) = \frac{P(\text{Spam}) \prod_{i=1}^{n} P(x_i \mid \text{Spam})}{P(X)}$$

Multinomial Naive Bayes computes class log-probabilities and calculates the confidence score.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure Python 3.8+ is installed on your system.

### 2. Install Dependencies
Run the following command in your terminal:
```bash
pip install -r requirements.txt
```

### 3. Run the Flask Web Application
```bash
python app.py
```
By default, the server starts at `http://127.0.0.1:5000/`.

---

## 📂 Project Structure
```
TSA2/
├── app.py                  # Flask backend server, model training & prediction endpoints
├── dataset.csv             # SMS dataset with labeled Spam and Ham messages
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
├── templates/
│   └── index.html          # Frontend UI HTML template
└── static/
    ├── css/
    │   └── style.css       # Lab UI styling & dynamic animations
    └── js/
        └── main.js         # Interactive JS script for AJAX requests & UI updates
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves main Web UI |
| `/api/info` | `GET` | Returns dataset size, vocabulary size, and model evaluation metrics |
| `/api/examples` | `GET` | Returns sample spam and legitimate messages for testing |
| `/predict` | `POST` | Accepts JSON `{ "message": "...", "do_lower": true, ... }` and returns classification result |

---

## 🧪 Sample API Request & Response

**POST `/predict`**
```json
{
  "message": "WINNER! You won a $1,000 Walmart Gift Card! Click http://bit.ly/claim-prize",
  "do_lower": true,
  "remove_punct": true,
  "remove_stop": true
}
```

**Response:**
```json
{
  "confidence": 98.45,
  "ham_probability": 1.55,
  "is_spam": true,
  "original_text": "WINNER! You won a $1,000 Walmart Gift Card! Click http://bit.ly/claim-prize",
  "prediction": "Spam",
  "preprocessed_text": "winner won 1 000 walmart gift card click http bit claim prize",
  "spam_probability": 98.45,
  "tokens": ["winner", "won", "1", "000", "walmart", "gift", "card", "click", "http", "bit", "claim", "prize"],
  "top_keywords": [
    { "bias": "spam", "tfidf": 0.38, "word": "winner" },
    { "bias": "spam", "tfidf": 0.35, "word": "claim" },
    { "bias": "spam", "tfidf": 0.32, "word": "prize" }
  ]
}
```
