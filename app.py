import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
import time

# Your emotion labels
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Emotion color mapping
emotion_colors = {
    "joy": "#FFD700", "anger": "#FF4500", "sadness": "#1E90FF",
    "surprise": "#BA55D3", "love": "#FF69B4", "fear": "#9370DB",
    "neutral": "#808080", "admiration": "#20B2AA", "amusement": "#FFA07A",
    "annoyance": "#CD853F", "approval": "#3CB371", "caring": "#FF6347",
    "confusion": "#778899", "curiosity": "#FF8C00", "desire": "#E9967A",
    "disappointment": "#4682B4", "disapproval": "#B22222", "disgust": "#6B8E23",
    "embarrassment": "#DB7093", "excitement": "#FF6347", "gratitude": "#2E8B57",
    "grief": "#483D8B", "nervousness": "#D2691E", "optimism": "#32CD32",
    "pride": "#4169E1", "realization": "#9932CC", "relief": "#00CED1",
    "remorse": "#8B4513"
}

@st.cache_resource
def load_model():
    try:
        model = AutoModelForSequenceClassification.from_pretrained("saved_model", local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained("saved_model", local_files_only=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None

# Custom CSS for enhanced UI
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stTextArea textarea { 
        border-radius: 10px; 
        padding: 15px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button { 
        border-radius: 20px; 
        background: linear-gradient(90deg, #FF6B6B 0%, #4ECDC4 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 10px 24px;
        width: 100%;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .emotion-card {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 8px 0 rgba(0,0,0,0.1);
        transition: 0.3s;
        border-left: 5px solid;
    }
    .emotion-card:hover {
        box-shadow: 0 8px 16px 0 rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    .header {
        font-size: 2.5em;
        color: #6a3093;
        text-align: center;
        margin-bottom: 20px;
        background: linear-gradient(90deg, #6a3093 0%, #a044ff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .dominant-emotion {
        font-size: 1.5em;
        font-weight: bold;
        text-align: center;
        padding: 15px;
        border-radius: 10px;
        margin: 20px 0;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="header">Emotion Detector</div>', unsafe_allow_html=True)
st.markdown("Discover the emotional spectrum in your text", unsafe_allow_html=True)

# Sidebar with info
with st.sidebar:
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-image: linear-gradient(#f8f9fa,white);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("### About")
    st.info("This app analyzes text for 28 different emotions using advanced AI.")
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Type or paste your text")
    st.markdown("2. Click 'Analyze Emotions'")
    st.markdown("3. Explore the results!")
    
    if st.checkbox("Show emotion legend"):
        st.markdown("### Emotion Color Guide")
        cols = st.columns(3)
        for i, (emotion, color) in enumerate(emotion_colors.items()):
            cols[i%3].markdown(f"<span style='color:{color}'>■</span> {emotion.capitalize()}", unsafe_allow_html=True)

# Main content
model, tokenizer, device = load_model()
if model is None:
    st.stop()

user_input = st.text_area("Share your thoughts here...", height=150, 
                         placeholder="Type something like: 'I just got the best news of my life!'")

if st.button("✨ Analyze Emotions"):
    if user_input:
        with st.spinner('Decoding emotions...'):
            time.sleep(1)  # For dramatic effect
            
            try:
                inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True).to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    probabilities = probabilities.cpu().numpy()[0]
                
                # Top emotion celebration
                top_emotion = emotion_labels[np.argmax(probabilities)]
                top_prob = np.max(probabilities)
                top_color = emotion_colors.get(top_emotion, "#6a3093")
                
                st.markdown(
                    f'<div class="dominant-emotion" style="background-color: {top_color}20; border: 2px solid {top_color}">'
                    f'Dominant Emotion: <span style="color: {top_color}">{top_emotion.upper()}</span> ({top_prob:.1%})'
                    f'</div>', 
                    unsafe_allow_html=True
                )
                
                # Emotion wheel visualization
                st.markdown("### Emotion Spectrum")
                cols = st.columns(6)
                for i, (emotion, prob) in enumerate(zip(emotion_labels, probabilities)):
                    if prob > 0.05:  # Only show significant emotions
                        color = emotion_colors.get(emotion, "#808080")
                        with cols[i%6]:
                            st.markdown(
                                f"""<div class="emotion-card" style="border-left-color: {color}">
                                    <b>{emotion.capitalize()}</b><br>
                                    {prob:.1%}
                                </div>""", 
                                unsafe_allow_html=True
                            )
                
                # Detailed analysis expander
                with st.expander("📊 Detailed Emotion Breakdown"):
                    for emotion, prob in zip(emotion_labels, probabilities):
                        color = emotion_colors.get(emotion, "#808080")
                        st.markdown(
                            f"""<div style="margin: 5px 0;">
                                <span style="font-weight:bold; width:120px; display:inline-block; color: {color}">
                                {emotion.capitalize()}</span>
                                <progress value="{prob}" max="1" style="width:300px; height:20px; accent-color: {color}"></progress>
                                <span style="margin-left:10px;">{prob:.1%}</span>
                            </div>""",
                            unsafe_allow_html=True
                        )
                
            except Exception as e:
                st.error(f"Oops! Something went wrong: {str(e)}")
    else:
        st.warning("Please enter some text to analyze")

# Footer
st.markdown("---")