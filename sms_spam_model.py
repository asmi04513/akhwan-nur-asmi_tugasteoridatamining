import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

# 1. Load data
df = pd.read_csv('spam.csv', encoding='latin-1')
df = df[['v1', 'v2']]
df.columns = ['label', 'text']

# 2. Explore
print(df['label'].value_counts())
print(df.isnull().sum())

# 3. Preprocess text simple
df['text_clean'] = df['text'].str.lower().str.replace(r'http\S+','', regex=True).str.replace(r'[^a-z\s]','', regex=True).str.strip()

# 4. Feature extraction
vect = TfidfVectorizer(ngram_range=(1,2), min_df=5, max_df=0.9)
X = vect.fit_transform(df['text_clean'])
y = np.where(df['label']=='spam', 1, 0)

# 5. Train‑test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# 6. Handle imbalance
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# 7. Define models & voting
clf_nb = MultinomialNB(alpha=1.0)
clf_lr = LogisticRegression(max_iter=1000, class_weight='balanced')
voting_clf = VotingClassifier(estimators=[('nb', clf_nb), ('lr', clf_lr)], voting='hard')

# 8. Train
voting_clf.fit(X_train_res, y_train_res)

# 9. Evaluate
y_pred = voting_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Akurasi test:", acc)
print(classification_report(y_test, y_pred, target_names=['Ham','Spam']))

# 10. Cross‑validation
cv_scores = cross_val_score(voting_clf, X_train_res, y_train_res, cv=5, scoring='accuracy')
print("CV Akurasi:", cv_scores.mean())

# 11. Save model & vectorizer
import pickle
with open('model/voting_sms_model.pkl', 'wb') as f:
    pickle.dump(voting_clf, f)
with open('model/vect_sms.pkl','wb') as f:
    pickle.dump(vect, f)

