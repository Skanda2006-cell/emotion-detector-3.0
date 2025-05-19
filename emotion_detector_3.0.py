import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import matplotlib.pyplot as plt
import base64

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
    "joy": "https://media3.giphy.com/media/5tkQ2D8oxYBVKwWNMV/giphy.gif",
    "sadness": "https://media1.giphy.com/media/mpy4YrE8nw6K3FCfz0/giphy.gif",
    "anger": "https://media2.giphy.com/media/69Egkd3vBh3AXuA5SC/giphy.gif",
    "fear": "https://media3.giphy.com/media/DHw6uxU2WbJ3a/giphy.gif",
    "surprise": "https://media1.giphy.com/media/E9bnFb6MvQgdTTS0CF/giphy.gif",
    "love": "https://media.giphy.com/media/l0HlOvJ7yaacpuSas/giphy.gif",
    "neutral": "https://media.giphy.com/media/xT9IgIc0lryrxvqVGM/giphy.gif",
    "disgust": "https://media.giphy.com/media/3o6Zt481isNVuQI1l6/giphy.gif"
}

# Page setup
st.set_page_config(page_title="Emotion Detector:text based 😎", layout="wide")
st.title("💬 Emotion Detector: Text based")
st.markdown("Multi-label detection + Reaction GIFs 😄🎬")

# Initialize mood diary
if 'mood_diary' not in st.session_state:
    st.session_state.mood_diary = []

# Load model
@st.cache_resource
def load_model():
    model_name = "j-hartmann/emotion-english-distilroberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)

classifier = load_model()

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

            top_emotion = sorted_results[0]['label']
            top_score = sorted_results[0]['score']

            # Show GIF
            top_gif = emotion_gifs.get(top_emotion)
            if top_gif:
                with col2:
                    gif_placeholder.image(top_gif, use_container_width=True)

            # --- Unified Vibe & Detection Display ---
            st.markdown("---")
            st.markdown("## 🎯 Emotion Meter + Top Emotions")

            # Top 1 Emotion - Bar and Confidence
            vibe_emoji = emotion_icons.get(top_emotion, "❓")
            vibe_color = emotion_colors.get(top_emotion, "#eee")
            st.markdown(f"<div style='background-color:{vibe_color}; padding:10px; border-radius:10px;'>"
                        f"<h3>{vibe_emoji} {top_emotion.capitalize()}</h3>"
                        f"<p style='font-size:18px;'>Confidence: {top_score:.2%}</p></div>",
                        unsafe_allow_html=True)
            st.progress(int(top_score * 100))

            # Top 3 Emotions
            st.markdown("### 🎭 Top 3 Detected Emotions")
            for r in sorted_results[:3]:
                label = r['label']
                score = r['score']
                emoji = emotion_icons.get(label, "")
                color = emotion_colors.get(label, "#eee")
                st.markdown(
                    f"<div style='background-color:{color}; padding:10px; border-radius:10px; font-size:18px;'>"
                    f"{emoji} <b>{label.capitalize()}</b>: {score:.2%}</div>",
                    unsafe_allow_html=True
                )

            st.session_state.mood_diary.append((text.strip(), top_emotion))

        except Exception as e:
            st.error(f"🔥 Error: {e}")

# Mood Diary Section
if st.session_state.mood_diary:
    st.markdown("---")
    st.subheader("📔 Mood Diary")
    for i, (entry, mood) in enumerate(st.session_state.mood_diary, 1):
        st.write(f"{i}. {entry} --> {mood}")

    # Download link
    txt_content = "Mood Diary - Emotion Detector\n" + "-"*40 + "\n"
    for i, (entry, mood) in enumerate(st.session_state.mood_diary, 1):
        txt_content += f"{i}. {entry} --> {mood}\n"

    b64 = base64.b64encode(txt_content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="mood_diary.txt">📥 Download Mood Diary (TXT)</a>'
    st.markdown(href, unsafe_allow_html=True)

# Pie Chart from Diary
if st.session_state.mood_diary:
    mood_count = {}
    for _, mood in st.session_state.mood_diary:
        mood_count[mood] = mood_count.get(mood, 0) + 1

    labels = list(mood_count.keys())
    sizes = list(mood_count.values())
    colors = plt.cm.Pastel1(range(len(labels)))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        colors=colors,
        startangle=140,
        textprops={'fontsize': 12}
    )
    ax.axis('equal')
    st.markdown("### 📊 Emotion Distribution (Pie Chart from Mood Diary)")
    st.pyplot(fig)

    with st.expander("ℹ️ About this app"):
        st.write("Powered by `j-hartmann/emotion-english-distilroberta-base` with multi-label emotion detection and GIF-based reactions.")
