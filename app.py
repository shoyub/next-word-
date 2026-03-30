'''
import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

#Load the LSTM Model
model=load_model('next_word_lstm.h5')

#3 Laod the tokenizer
with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

# Function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]  # Ensure the sequence length matches max_sequence_len-1
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
    predicted = model.predict(token_list, verbose=0)
    predicted_word_index = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None

# streamlit app
st.title("Next Word Prediction With LSTM And Early Stopping")
input_text=st.text_input("Enter the sequence of Words","To be or not to")
if st.button("Predict Next Word"):
    max_sequence_len = model.input_shape[1] + 1  # Retrieve the max sequence length from the model input shape
    next_word = predict_next_word(model, tokenizer, input_text, max_sequence_len)
    st.write(f'Next word: {next_word}')

'''
import streamlit as st
import numpy as np
import pickle
from tf_keras.models import load_model
from tf_keras.preprocessing.sequence import pad_sequences

# Load both models
lstm_model = load_model('next_word_lstm.h5')
gru_model = load_model('gru_model.h5')

# Load tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Reverse index (fast lookup)
index_word = {index: word for word, index in tokenizer.word_index.items()}

# ---- OPTIONAL: Hardcode metrics (BEST PRACTICE for Streamlit) ----
lstm_acc = 0.85   # replace with your actual value
lstm_loss = 1.20

gru_acc = 0.83    # replace with your actual value
gru_loss = 1.30

# Prediction function
def predict_next_word(model, tokenizer, text, max_sequence_len):
    token_list = tokenizer.texts_to_sequences([text])[0]

    if len(token_list) >= max_sequence_len:
        token_list = token_list[-(max_sequence_len-1):]

    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')

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
    st.write(" LSTM:", lstm_pred)
    st.write(" GRU:", gru_pred)

    # ---- Metrics Section ----
    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("LSTM Accuracy", f"{lstm_acc:.4f}")
        st.metric("LSTM Loss", f"{lstm_loss:.4f}")

    with col2:
        st.metric("GRU Accuracy", f"{gru_acc:.4f}")
        st.metric("GRU Loss", f"{gru_loss:.4f}")

    # ---- Best Model ----
    if lstm_acc > gru_acc:
        st.success(" LSTM performs better")
    else:
        st.success(" GRU performs better")