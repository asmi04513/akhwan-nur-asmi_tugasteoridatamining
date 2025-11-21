import os
import pickle
import streamlit as st
import pickle
import numpy as np

# Load model & vectorizer
with open('model/vect_sms.pkl','rb') as f:
    vect = pickle.load(f)
with open('model/voting_sms_model.pkl','rb') as f:
    model = pickle.load(f)

st.title("SMS Spam Detector")
st.write("Masukkan pesan SMS Anda dan lihat prediksinya (ham = legit, spam = tidak diinginkan).")

user_input = st.text_area("Pesan SMS:")

if st.button("Prediksi"):
    cleaned = user_input.lower()
    cleaned = ''.join(ch for ch in cleaned if ch.isalpha() or ch.isspace())
    X_input = vect.transform([cleaned])
    pred = model.predict(X_input)[0]
    label = "Spam" if pred==1 else "Ham (Legit)"
    st.write("Hasil prediksi: **{}**".format(label))

    # tidak semua model voting punya predict_proba; jika punya:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_input)[0][1]
        st.write(f"Probabilitas spam: {proba:.2%}")

# Sidebar – performansi model
st.sidebar.header("Performansi Model")
st.sidebar.write("Akurasi test: {:.2%}".format(acc))  # acc dari kode training
# Buat folder 'model' jika belum ada
os.makedirs('model', exist_ok=True)

# Simpan model dan vectorizer
with open('model/voting_sms_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)

with open('model/vect_sms.pkl', 'wb') as f:
    pickle.dump(vect, f)

print("Model dan vectorizer berhasil disimpan di folder 'model/'")
