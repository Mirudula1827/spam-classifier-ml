import streamlit as st
import pandas as pd
import string
import time
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Spam Classifier", layout="centered")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

/* === Global === */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .block-container {
    background-color: #EFE8DE !important;
    color: #1A1A1A !important;
    font-family: 'DM Sans', sans-serif !important;
}
.block-container {
    max-width: 700px !important;
    padding-top: 2.8rem !important;
    padding-bottom: 3rem !important;
}

/* === Title === */
.app-title {
    text-align: center;
    font-family: 'Playfair Display', serif;
    font-size: 4.2rem;
    font-weight: 700;
    color: #1A1A1A;
    margin-bottom: 0.2rem;
    line-height: 1.15;
}
.app-title span { color: #9E7B52; }
.app-subtitle {
    text-align: center;
    font-size: 1.2rem;
    color: #9E7B52;
    margin-bottom: 2rem;
    letter-spacing: 0.3px;
}

/* === Section headings === */
.section-heading {
    font-size: 0.95rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #9E7B52;
    font-weight: 600;
    margin-top: 2rem;
    margin-bottom: 0.7rem;
}

/* === Textarea === */
div[data-testid="stTextArea"] label { display: none !important; }
div[data-testid="stTextArea"] textarea {
    background-color: #FAF6F0 !important;
    border: 1.5px solid #D4C9B8 !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #1A1A1A !important;
    padding: 0.9rem 1.1rem !important;
    box-shadow: none !important;
    resize: vertical !important;
}
div[data-testid="stTextArea"] textarea::placeholder { color: #B5A898 !important; }
div[data-testid="stTextArea"] textarea:focus {
    border-color: #9E7B52 !important;
    box-shadow: 0 0 0 3px rgba(158,123,82,0.13) !important;
}

/* === Sample pills === */
.sample-label {
    font-size: 0.95rem;
    color: #9E7B52;
    margin-bottom: 0.4rem;
    letter-spacing: 0.5px;
}

/* === Run button === */
div[data-testid="stButton"] > button {
    background-color: #1A1A1A !important;
    color: #EFE8DE !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    padding: 0.58rem 2.4rem !important;
    width: 100% !important;
    transition: background 0.2s;
    box-shadow: none !important;
    letter-spacing: 0.3px;
    white-space: nowrap;
    height: 48px;
    min-width: 120px;
}
div[data-testid="stButton"] > button:hover { background-color: #2e2e2e !important; }
div[data-testid="stButton"] > button:disabled {
    background-color: #C5B9AA !important;
    color: #EFE8DE !important;
    cursor: not-allowed !important;
}

/* === Result card === */
.result-card {
    background: #FAF6F0;
    border: 1.5px solid #D4C9B8;
    border-radius: 12px;
    padding: 0.9rem 1.15rem;
    margin-bottom: 0.55rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    transition: box-shadow 0.18s ease, border-color 0.18s ease;
}
.result-card:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.07);
    border-color: #B8A898;
}
.result-card .msg-text {
    font-size: 0.87rem;
    color: #2C2C2C;
    flex: 1;
    line-height: 1.5;
}
.badge {
    padding: 0.25rem 0.85rem;
    border-radius: 20px;
    font-size: 0.73rem;
    font-weight: 500;
    white-space: nowrap;
    flex-shrink: 0;
    letter-spacing: 0.2px;
}
.badge-spam { background: #F0D8C4; color: #7A3A0A; }
.badge-ham  { background: #CDE8CC; color: #165416; }

/* === Metric cards (native st.metric override) === */
[data-testid="stMetric"] {
    background: #FAF6F0 !important;
    border: 1.5px solid #D4C9B8 !important;
    border-radius: 12px !important;
    padding: 1rem 1.1rem !important;
    text-align: center !important;
    transition: box-shadow 0.18s ease, border-color 0.18s ease !important;
}
[data-testid="stMetric"]:hover {
    box-shadow: 0 4px 16px rgba(0,0,0,0.07) !important;
    border-color: #B8A898 !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.62rem !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase !important;
    color: #9E7B52 !important;
    font-weight: 600 !important;
    justify-content: center !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Playfair Display', serif !important;
    font-size: 1.55rem !important;
    color: #1A1A1A !important;
    font-weight: 700 !important;
    justify-content: center !important;
}
[data-testid="stMetricDelta"] { display: none !important; }

/* === Divider === */
hr {
    border: none !important;
    border-top: 1px solid #D4C9B8 !important;
    margin: 1.8rem 0 !important;
}

/* === Warning / info === */
[data-testid="stAlert"] {
    background-color: #E8DFCF !important;
    color: #5A4020 !important;
    border: 1px solid #D4C9B8 !important;
    border-radius: 10px !important;
    font-size: 0.85rem !important;
}

/* === Footer === */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #5A4A3A;
    margin-top: 2.5rem;
    letter-spacing: 0.3px;
}

/* === Hide Streamlit chrome === */
#MainMenu, footer, header { visibility: hidden !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = ''.join(c for c in text if c not in string.punctuation)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

def clean_series(texts):
    if isinstance(texts, list):
        texts = pd.Series(texts)
    return texts.apply(clean_text)


@st.cache_resource(show_spinner=False)
def load_and_train():
    try:
        df = pd.read_csv('spam.csv', encoding='latin-1')
        df = df[['v1', 'v2']]
        df.columns = ['label', 'text']
        df['label'] = df['label'].str.strip().str.lower().map({'ham': 0, 'spam': 1})
        df.drop_duplicates(inplace=True)
        df.dropna(inplace=True)
    except FileNotFoundError:
        data = {
            'label': [1,1,1,1,1,1,0,0,0,0,0,0,1,0,1,0,1,0],
            'text': [
                "URGENT You have won a 1 lakh cash prize call now",
                "Free entry claim your reward immediately",
                "Limited offer just for you click now",
                "Win money now click here",
                "Exclusive deal available today only claim now",
                "Congratulations you are selected for cash prize",
                "Hey are we meeting tomorrow",
                "I will call you later",
                "Your OTP is 458921 do not share it",
                "Amazon order has been shipped successfully",
                "Lunch at 1pm works for me",
                "Can you send me the report by EOD",
                "You have been chosen for a free gift",
                "Let me know if you need anything else",
                "Click here to claim your free vacation package",
                "See you at the office tomorrow morning",
                "Dear customer your account has been selected for a reward",
                "Happy birthday hope you have a great day",
            ]
        }
        df = pd.DataFrame(data)

    X, y = df['text'], df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

    model = make_pipeline(
        FunctionTransformer(clean_series),
        TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95),
        LogisticRegression(max_iter=1000, class_weight='balanced')
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    perf = {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1 Score':  f1_score(y_test, y_pred, zero_division=0),
    }
    return model, perf


# ── Train ─────────────────────────────────────────────────────────────────────
with st.spinner("Loading model…"):
    model, perf = load_and_train()

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="app-title">Spam <span>Classifier.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Detect spam messages instantly using machine learning</div>', unsafe_allow_html=True)

# ── Sample messages ───────────────────────────────────────────────────────────
SAMPLES = {
    "💬 Casual":    "Hey, are we still on for lunch tomorrow?",
    "🚨 Promo":     "URGENT! You've won a cash prize. Call now to claim!",
    "📦 Transact":  "Your Amazon order has been shipped successfully.",
    "🎁 Bait":      "Free entry! Click now to claim your exclusive reward.",
}

st.markdown('<div class="section-heading">✦ Input</div>', unsafe_allow_html=True)
st.markdown('<div class="sample-label">Try a sample →</div>', unsafe_allow_html=True)

cols = st.columns(len(SAMPLES))
for col, (label, msg) in zip(cols, SAMPLES.items()):
    with col:
        if st.button(label, key=f"sample_{label}"):
            st.session_state["input_box"] = msg

# ── Text input ────────────────────────────────────────────────────────────────

# Initialize once
if "input_box" not in st.session_state:
    st.session_state["input_box"] = ""

user_input = st.text_area(
    label="messages",
    placeholder="Type or paste messages here — one per line…",
    height=130,
    label_visibility="hidden",
    key="input_box"   # ✅ ONLY key, NO value
)

run_clicked = st.button("Check", key="run_btn")

# ── Results ───────────────────────────────────────────────────────────────────
if run_clicked and user_input.strip():
    messages = [m.strip() for m in user_input.strip().splitlines() if m.strip()]

    with st.spinner("Classifying…"):
        time.sleep(0.3)  # brief UX pause so spinner is visible
        preds = model.predict(messages)
        probas = model.predict_proba(messages)

    st.markdown("---")
    st.markdown('<div class="section-heading">✦ Results</div>', unsafe_allow_html=True)

    for msg, pred, proba in zip(messages, preds, probas):
        confidence = int(round(max(proba) * 100))
        if pred == 1:
            badge = f'<span class="badge badge-spam">🚨 Spam ({confidence}%)</span>'
        else:
            badge = f'<span class="badge badge-ham">✅ Not Spam ({confidence}%)</span>'

        st.markdown(f"""
        <div class="result-card">
            <span class="msg-text">{msg}</span>
            {badge}
        </div>
        """, unsafe_allow_html=True)

elif run_clicked:
    st.warning("Please enter at least one message before running.")

# ── Model performance ─────────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-heading">✦ Model Performance</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
for col, (name, val) in zip([c1, c2, c3, c4], perf.items()):
    with col:
        st.metric(label=name, value=f"{val:.2%}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Built with Streamlit · Logistic Regression · TF-IDF</div>',
    unsafe_allow_html=True
)
