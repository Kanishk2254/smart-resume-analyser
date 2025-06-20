import pandas as pd
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Load DataSet

df = pd.read_csv("Resume.csv") # or the correct path to your CSV
df = df[['Category', 'Resume_str']].dropna()

# Clean Text
def clean_text(text):
    text=text.lower()
    text=re.sub(r"https\\S+ | www\\S+","",text)
    text.translate(str.maketrans('','',string.punctuation))
    text=re.sub(r'[^a-zA-Z\s]','',text)
    text=re.sub(r'\\s+','',text.strip())
    return text

df['cleaned_resume']=df['Resume_str'].apply(clean_text)

# TF-IDF Vectorizer
vectorizer=TfidfVectorizer(max_features=3000)
X=vectorizer.fit_transform(df['cleaned_resume'])

# Labels
y = df['Category']

# Train-Test Split
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluation
y_pred= clf.predict(X_test)
print("Accuracy: ",accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))

# Save
joblib.dump(clf,'job_model.pkl')
joblib.dump(vectorizer,'vectorizer.pkl')
print("Model and Vectorizer saved Succsefully.")