import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Emojis, colors, and GIFs
emotion_icons = {
    "joy": "😄", "sadness": "😢", "anger": "😠", "fear": "😱",
    "surprise": "😲", "love": "❤️", "neutral": "😐", "disgust": "🤢"
}
emotion_colors = {
    "joy": "#fff9c4", "sadness": "#bbdefb", "anger": "#ffcdd2",
    "fear": "#d1c4e9", "surprise": "#e1bee7", "love": "#f8bbd0",
    "neutral": "#cfd8dc", "disgust": "#dcedc8"
}
emotion_gifs = {
    "joy": "https://media.giphy.com/media/5GoVLqeAOo6PK/giphy.gif",
    "sadness": "https://media.giphy.com/media/d2lcHJTG5Tscg/giphy.gif",
    "anger": "https://media.giphy.com/media/l3V0j3ytFyGHqiV7W/giphy.gif",
    "fear": "https://media3.giphy.com/media/v1.Y2lkPTc5MGI3NjExMmsyOGRqMm1lcXg3Zjd5bW9uc3hzbno0YmJodDR3MGszdHRwbTZ0dCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/DHw6uxU2WbJ3a/giphy.gif",
    "surprise": "https://media.giphy.com/media/l0MYt5jPR6QX5pnqM/giphy.gif",
    "love": "https://media.giphy.com/media/l0HlOvJ7yaacpuSas/giphy.gif",
    "neutral": "https://media.giphy.com/media/xT9IgIc0lryrxvqVGM/giphy.gif",
    "disgust": "https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif"
}

# Page setup
st.set_page_config(page_title="Emotion Detector 3.0 😎", layout="wide")
st.title("💬 Emotion Detector 3.0")
st.markdown("Multi-label detection + Reaction GIFs 😄🎬")

# Load model
@st.cache_resource
def load_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

classifier = load_model()

# Initialize mood diary
if "mood_diary" not in st.session_state:
    st.session_state.mood_diary = []

# Layout with columns: input + GIF
col1, col2 = st.columns([2, 1])

with col1:
    text = st.text_area("Type your sentence 👇", height=180)

with col2:
    st.markdown("### 🎬 Reaction Preview")
    gif_placeholder = st.empty()

# Analyze button
if st.button("Analyze 🔍"):
    if not text.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        try:
            results = classifier(text.strip())[0]
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

            threshold = 0.1
            detected = [r for r in sorted_results if r['score'] >= threshold]

            if not detected:
                st.warning("😕 No strong emotion detected. Try something more expressive.")
            else:
                top_emotion = sorted_results[0]['label']
                st.session_state.mood_diary.append({"sentence": text.strip(), "emotion": top_emotion})

                top_gif = emotion_gifs.get(top_emotion)
                if top_gif:
                    with col2:
                        gif_placeholder.image(top_gif, use_container_width=True)

                st.markdown("---")
                st.subheader("🎭 Detected Emotions")
                for r in detected:
                    label = r['label']
                    score = r['score']
                    emoji = emotion_icons.get(label, "")
                    color = emotion_colors.get(label, "#eee")
                    st.markdown(
                        f"<div style='background-color:{color}; padding:10px; border-radius:10px; font-size:18px;'>"
                        f"{emoji} <b>{label.capitalize()}</b>: {score:.2%}</div>",
                        unsafe_allow_html=True
                    )

        except Exception as e:
            st.error(f"🔥 Error: {e}")

# Show Mood Diary
if st.session_state.mood_diary:
    st.markdown("### 📘 Mood Diary")
    st.dataframe(st.session_state.mood_diary[::-1])

    # Combined pie chart
    emotion_counter = {}
    for entry in st.session_state.mood_diary:
        emotion = entry["emotion"]
        emotion_counter[emotion] = emotion_counter.get(emotion, 0) + 1

    labels = list(emotion_counter.keys())
    sizes = list(emotion_counter.values())
    colors = plt.cm.Pastel1(range(len(labels)))

    fig, ax = plt.subplots(figsize=(4, 4))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        textprops={'fontsize': 12}
    )
    ax.axis('equal')
    st.markdown("### 📊 Emotion Distribution from Mood Diary")
    st.pyplot(fig)

with st.expander("ℹ️ About this app"):
    st.write("Powered by `j-hartmann/emotion-english-distilroberta-base` with multi-label emotion detection and GIF-based reactions. Includes mood diary and cumulative emotion analytics.")
