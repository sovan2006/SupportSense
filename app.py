import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(
    page_title="SupportSense AI",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: 800;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #f1f5f9;
    border: 1px solid #e2e8f0;
    margin-top: 20px;
}

.intent {
    font-size: 24px;
    font-weight: 700;
}

.confidence {
    font-size: 18px;
    margin-top: 8px;
}

.footer {
    text-align: center;
    color: #94a3b8;
    margin-top: 40px;
    font-size: 14px;
}

</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_supportsense_model():
    model = tf.keras.models.load_model(
        "SupportSense_best_model.keras"
    )

    with open("SupportSense_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("SupportSense_label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    with open("SupportSense_config.pkl", "rb") as f:
        config = pickle.load(f)

    return model, tokenizer, label_encoder, config

try:
    model, tokenizer, label_encoder, config = load_supportsense_model()

except Exception as e:
    st.error("Unable to load the SupportSense model files.")
    st.code(str(e))
    st.stop()

MAX_LENGTH = config["max_length"]

def predict_intent(text):
    sequence = tokenizer.texts_to_sequences([text])

    padded_sequence = pad_sequences(
        sequence,
        maxlen=MAX_LENGTH,
        padding="post",
        truncating="post"
    )

    probabilities = model.predict(
        padded_sequence,
        verbose=0
    )

    predicted_class = np.argmax(probabilities[0])

    predicted_intent = label_encoder.inverse_transform(
        [predicted_class]
    )[0]

    confidence = float(
        probabilities[0][predicted_class]
    )

    return predicted_intent, confidence

st.markdown(
    '<div class="title">🤖 SupportSense AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent Customer Support Intent Classification'
    '</div>',
    unsafe_allow_html=True
)

with st.expander("ℹ️ About SupportSense"):
    st.write("""
    **SupportSense AI** is an NLP-based customer support
    intent classification system.

    It analyzes a customer's query and predicts the most
    relevant support intent using a trained neural-network
    model.
    """)

    st.write(
        f"**Number of Intent Classes:** "
        f"{len(label_encoder.classes_)}"
    )

    st.write(
        f"**Maximum Sequence Length:** "
        f"{MAX_LENGTH}"
    )

st.subheader("💬 Ask SupportSense")

user_query = st.text_area(
    "Enter your customer support query:",
    placeholder="Example: Why was my card payment declined?",
    height=120
)

if st.button(
    "🔍 Predict Intent",
    use_container_width=True
):

    if not user_query.strip():
        st.warning("Please enter a customer query.")

    else:
        with st.spinner("Analyzing customer query..."):
            intent, confidence = predict_intent(user_query)

        st.markdown(
            '<div class="result-box">',
            unsafe_allow_html=True
        )

        st.markdown("### 🎯 Predicted Intent")

        st.markdown(
            f'<div class="intent">{intent}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f'<div class="confidence">'
            f'Confidence: {confidence:.2%}'
            f'</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

        st.write("### 📊 Prediction Confidence")

        st.progress(min(confidence, 1.0))

st.write("---")

st.subheader("💡 Try an Example")

examples = [
    "Why was my card payment declined?",
    "I want to cancel my transfer",
    "I don't recognize this transaction",
    "How can I change my PIN?",
    "My cash withdrawal failed"
]

for example in examples:

    if st.button(
        example,
        use_container_width=True
    ):

        intent, confidence = predict_intent(example)

        st.success(
            f"Intent: **{intent}**"
        )

        st.info(
            f"Confidence: **{confidence:.2%}**"
        )

st.markdown(
    '<div class="footer">'
    'SupportSense AI • NLP Customer Support Classification'
    '</div>',
    unsafe_allow_html=True
)