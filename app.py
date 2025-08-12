import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# ==============================
# 📌 Page Config
# ==============================
st.set_page_config(
    page_title="📩 SMS Spam Classifier",
    layout="centered",
    page_icon="📩",
    initial_sidebar_state="collapsed"
)

# ==============================
# 📌 Downloads (Run only once per environment)
# ==============================
nltk.download('stopwords')
nltk.download('punkt')

# ==============================
# 📌 Preprocessing
# ==============================
ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    # Remove non-alphanumeric tokens
    text = [i for i in text if i.isalnum()]

    # Remove stopwords & punctuation
    text = [i for i in text if i not in stopwords.words('english') and i not in string.punctuation]

    # Stemming
    text = [ps.stem(i) for i in text]

    return " ".join(text)

# ==============================
# 📌 Load Models
# ==============================
tfidf = pickle.load(open(r"C:\Users\akhil\sms-class\vectorizer (1).pkl", 'rb'))
model = pickle.load(open(r"C:\Users\akhil\sms-class\model (1).pkl", 'rb'))

# ==============================
# 🎨 Custom CSS
# ==============================
st.markdown("""
    <style>
        .main {
            background-color: #f4f6f8;
        }
        h1 {
            text-align: center;
            color: #2c3e50;
        }
        .stTextInput>div>div>input {
            background-color: white;
            color: black;
        }
        .result-box {
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
            padding: 15px;
            border-radius: 10px;
        }
        .spam {
            background-color: #ffcccc;
            color: #b30000;
        }
        .not-spam {
            background-color: #ccffcc;
            color: #006600;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================
# 📌 UI
# ==============================
st.title("📩 SMS Spam Classifier")
st.write("Detect whether a message is **Spam** or **Not Spam** instantly!")

input_sms = st.text_input("✉️ Enter the message:")

if st.button("🔍 Predict"):
    if input_sms.strip() == "":
        st.warning("⚠️ Please enter a message before predicting.")
    else:
        # Preprocess
        transformed_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transformed_sms])
        result = model.predict(vector_input)[0]

        # Display result with styled box
        if result == 1:
            st.markdown('<div class="result-box spam">🚫 Spam</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-box not-spam">✅ Not Spam</div>', unsafe_allow_html=True)
