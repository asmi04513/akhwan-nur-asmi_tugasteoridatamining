import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="SMS Spam Detector", page_icon="📩", layout="centered")

# -------------------------------
# 1️⃣ Buat folder 'model' jika belum ada
os.makedirs('model', exist_ok=True)

st.title("SMS Spam Detection App")
st.write("Masukkan SMS untuk memprediksi apakah spam atau ham.")

# -------------------------------
# 2️⃣ Cek apakah dataset spam.csv ada
dataset_path = 'spam.csv'
if not os.path.exists(dataset_path):
    st.error("File 'spam.csv' tidak ditemukan di folder proyek!")
    st.stop()

# -------------------------------
# 3️⃣ Load dataset
df = pd.read_csv(dataset_path, encoding='latin-1')
df = df[['v1','v2']]
df.columns = ['label','text']
df['label'] = df['label'].map({'ham':0,'spam':1})

# -------------------------------
# 4️⃣ Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -------------------------------
# 5️⃣ TF-IDF Vectorization
vect = TfidfVectorizer(stop_words='english')
X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)

# -------------------------------
# 6️⃣ Handle imbalance dengan SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_vect, y_train)

# -------------------------------
# 7️⃣ Definisikan dua model
model_nb = MultinomialNB()
model_lr = LogisticRegression(max_iter=1000)
voting_clf = VotingClassifier(estimators=[('NB', model_nb), ('LR', model_lr)], voting='hard')
voting_clf.fit(X_train_res, y_train_res)

# -------------------------------
# 8️⃣ Evaluasi model
y_pred = voting_clf.predict(X_test_vect)
acc = accuracy_score(y_test, y_pred)
st.success(f"Model terlatih dengan akurasi: {acc*100:.2f}%")

# -------------------------------
# 9️⃣ Simpan model & vectorizer
with open('model/voting_sms_model.pkl','wb') as f:
    pickle.dump(voting_clf, f)
with open('model/vect_sms.pkl','wb') as f:
    pickle.dump(vect, f)

st.info("Model dan vectorizer tersimpan di folder 'model/'")

# -------------------------------
# 10️⃣ Prediksi SMS
msg_input = st.text_area("Ketik SMS di sini:")

if st.button("Prediksi"):
    if msg_input.strip() == "":
        st.warning("Silakan masukkan pesan terlebih dahulu.")
    else:
        msg_vect = vect.transform([msg_input])
        pred = voting_clf.predict(msg_vect)[0]
        result = "SPAM 🚫" if pred==1 else "HAM ✅"
        st.success(f"Hasil Prediksi: {result}")
