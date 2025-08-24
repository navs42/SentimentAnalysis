import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import torch.nn.functional as F
import pandas as pd


EMOTIONS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval", "disgust",
    "embarrassment", "excitement", "fear", "gratitude", "grief", "joy",
    "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral"
]


@st.cache_resource
def load_model():
    try:
        # Try local first
        model = AutoModelForSequenceClassification.from_pretrained("saved_model", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained("saved_model", local_files_only=True)
        st.success("✅ Loaded model from local folder")
    except:
        # Fallback to Hugging Face
        model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
        st.warning("⚠️ Local model not found, downloading from Hugging Face...")
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Save locally for next time (offline use)
        model.save_pretrained("saved_model")
        tokenizer.save_pretrained("saved_model")
        st.info("📂 Model saved locally for future runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    return model, tokenizer, device


def predict_emotions(text, model, tokenizer, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    return probs

def local_css():
    st.markdown(
        """
        <style>
        .emotion-card {
            padding: 12px;
            margin: 8px 0;
            border-radius: 10px;
            font-weight: bold;
            color: white;
            text-align: center;
        }
        .dominant {
            border: 2px solid #FFD700;
            box-shadow: 0 0 12px #FFD700;
        }
        .progress-bar {
            height: 20px;
            border-radius: 10px;
        }
        .hover:hover {
            transform: scale(1.02);
            transition: 0.3s;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="Emotion Detector", page_icon="😊", layout="centered")
    local_css()

    st.title("🎭 Emotion Detector")
    st.subheader("Discover the emotional spectrum in your text")

    # Load model
    with st.spinner("🔄 Loading model..."):
        model, tokenizer, device = load_model()

    # Text input
    user_input = st.text_area("✍️ Enter your text here:", height=150)

    if st.button("🔍 Analyze Emotion"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text to analyze.")
        else:
            with st.spinner("🔎 Analyzing..."):
                probs = predict_emotions(user_input, model, tokenizer, device)
                df = pd.DataFrame({"Emotion": EMOTIONS, "Probability": probs})
                df = df.sort_values("Probability", ascending=False).reset_index(drop=True)

                # Dominant emotion
                top_emotion = df.iloc[0]

                st.success(f"✨ **Dominant Emotion:** {top_emotion['Emotion'].capitalize()} "
                           f"({top_emotion['Probability']*100:.2f}%)")

                # Show all emotions
                st.markdown("### 📊 Emotion Probabilities")
                for idx, row in df.iterrows():
                    progress = int(row["Probability"] * 100)
                    bar_color = "#4CAF50" if idx == 0 else "#1E90FF"
                    css_class = "emotion-card hover dominant" if idx == 0 else "emotion-card hover"

                    st.markdown(
                        f"""
                        <div class="{css_class}" style="background-color:{bar_color};">
                            {row['Emotion'].capitalize()} - {progress}%
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(row["Probability"])

if __name__ == "__main__":
    main()
