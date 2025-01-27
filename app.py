import streamlit as st
import joblib
import re

# Load the trained model and vectorizer
model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Function to clean text
def clean_text(text):
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

# Streamlit UI
st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection App")
st.write("### Enter a news article below to check if it's **real** or **fake.**")

# User input
user_input = st.text_area("📝 Paste the news text here:", height=200)

if st.button("🔍 Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text to predict.")
    else:
        # Preprocess input text
        cleaned_input = clean_text(user_input)
        input_tfidf = vectorizer.transform([cleaned_input])

        # Predict using the model
        prediction = model.predict(input_tfidf)[0]

        # Display result
        st.subheader("🔎 Prediction Result")
        st.success(f"**Prediction:** {'🟢 Real' if prediction == 1 else '🔴 Fake'}")

# Footer
st.markdown("---")
st.markdown("#### Made with ❤️ by Chetan Shakya")
