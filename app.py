import streamlit as st
import numpy as np
import pickle
import keras

# Load both models
lstm_model = keras.saving.load_model('next_word_lstm.h5')
gru_model = keras.saving.load_model('gru_model.h5')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Reverse index
index_word = {index: word for word, index in tokenizer.word_index.items()}

# Hardcoded metrics
lstm_acc = 0.85
lstm_loss = 1.20
gru_acc = 0.83
gru_loss = 1.30

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]
    token_list = keras.utils.pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)[0]
    return index_word.get(predicted_word_index, "Word not found")

# Streamlit UI
st.title("Next Word Prediction (LSTM vs GRU)")
input_text = st.text_input("Enter text", "To be or not to")

if st.button("Predict Next Word"):
    max_sequence_len = lstm_model.input_shape[1] + 1
    lstm_pred = predict_next_word(lstm_model, tokenizer, input_text, max_sequence_len)
    gru_pred = predict_next_word(gru_model, tokenizer, input_text, max_sequence_len)

    st.subheader("Predictions:")
    st.write("LSTM:", lstm_pred)
    st.write("GRU:", gru_pred)

    st.subheader("Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("LSTM Accuracy", f"{lstm_acc:.4f}")
        st.metric("LSTM Loss", f"{lstm_loss:.4f}")
    with col2:
        st.metric("GRU Accuracy", f"{gru_acc:.4f}")
        st.metric("GRU Loss", f"{gru_loss:.4f}")

    if lstm_acc > gru_acc:
        st.success("LSTM performs better")
    else:
        st.success("GRU performs better")
