import streamlit as st
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import pickle

vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])  # Tiến hành stemming
    return text


st.title('Email Spam Classifier')
message = st.text_input('Input message')


if st.button('Predict'):
    processed_message = preprocess_text(message)
    message_vect = vectorizer.transform([message])
    prediction = model.predict(message_vect)
    st.write("Prediction output:", prediction)
    if int(prediction[0]) == 1:
        st.warning('This message is spam')
    else:
        st.success('This message is ham')


