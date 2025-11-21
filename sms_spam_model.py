# sms_spam_model.py
import pandas as pd
import numpy as np
import os
import pickle

# Algoritma ML
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

# 1. Buat folder 'model' jika belum ada
os.makedirs('model', exist_ok=True)

# 2. Load dataset
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# 3. Encode label: ham=0, spam=1
df['label'] = df['label'].map({'ham':0, 'spam':1})

# 4. Split data
X = df['text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 5. TF-IDF Vectorization
vect = TfidfVectorizer(stop_words='english')
X_train_vect = vect.fit_transform(X_train)
X_test_vect = vect.transform(X_test)

# 6. Handle imbalance dengan SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train_vect, y_train)

# 7. Definisikan dua model
model_nb = MultinomialNB()
model_lr = LogisticRegression(max_iter=1000)

# 8. Voting Classifier (hard voting)
voting_clf = VotingClassifier(estimators=[('NB', model_nb), ('LR', model_lr)], voting='hard')
voting_clf.fit(X_train_res, y_train_res)

# 9. Evaluasi model
y_pred = voting_clf.predict(X_test_vect)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy VotingClassifier: {acc*100:.2f}%")
print(classification_report(y_test, y_pred))

# 10. Simpan model & vectorizer
with open('model/voting_sms_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)

with open('model/vect_sms.pkl','wb') as f:
    pickle.dump(vect, f)

print("Model dan vectorizer berhasil disimpan di folder 'model/'")
