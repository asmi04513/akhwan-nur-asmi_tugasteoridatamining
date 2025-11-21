# app_streamlit.py
import streamlit as st
import pickle
import numpy as np

# 1. Load model & vectorizer
with open('model/voting_sms_model.pkl','rb') as f:
    model = pickle.load(f)

with open('model/vect_sms.pkl','rb') as f:
    vect = pickle.load(f)

# 2. Streamlit title & deskripsi
st.title("SMS Spam Detection")
st.write("Masukkan pesan SMS, aplikasi ini akan memprediksi apakah spam atau ham.")

# 3. Input form
msg_input = st.text_area("Ketik SMS di sini:")

# 4. Prediksi
if st.button("Prediksi"):
    if msg_input.strip() == "":
        st.warning("Silakan masukkan pesan terlebih dahulu.")
    else:
        msg_vect = vect.transform([msg_input])
        pred = model.predict(msg_vect)[0]
        result = "SPAM 🚫" if pred == 1 else "HAM ✅"
        st.success(f"Hasil Prediksi: {result}")
