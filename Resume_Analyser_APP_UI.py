import streamlit as st 
import joblib
import PyPDF2
from PyPDF2 import PdfReader
from PyPDF2 import PdfMerger
import re
import string
from sklearn.preprocessing import LabelEncoder
import numpy as np
from fpdf import FPDF
import tempfile

# Load Model and Vectorizer
model=joblib.load('job_model.pkl')
vectorizer=joblib.load('vectorizer.pkl')

# Clean Text
def clean_text(text):
    text=text.lower()
    text=re.sub(r"https\\S+ | www\\S+","",text)
    text.translate(str.maketrans('','',string.punctuation))
    text=re.sub(r'[^a-zA-Z\s]','',text)
    text=re.sub(r'\\s+','',text.strip())
    return text

# Extract PDF
def extract_text_from_pdf(upload_file):
    pdf_reader = PyPDF2.PdfReader(upload_file)
    text=""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Highlight Keywords
def highlight_keywords(text, top_words):
    highlighted = text
    for word in top_words:
        highlighted = re.sub(f'(?!)({word})',r'**\1**',highlighted)
    return highlighted

# Skill Categotization
TECH_SKILLS=['python', 'java', 'c++', 'sql', 'pandas', 'excel', 'deep learning', 'machine learning', 'linux']
SOFT_SKILLS=['communication', 'leadership', 'teamwork', 'creativity', 'problem solving', 'adaptability']

def categorize_skills(text):
    found_tech = [skill for skill in TECH_SKILLS if skill in text.lower()]
    found_soft = [skill for skill in SOFT_SKILLS if skill in text.lower()]
    return found_tech, found_soft

# PDF Report Generator
def generate_pdf_report(prediction, top_words, tech, soft, match_score = None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size = 12)
    
    pdf.cell(200, 10, txt = "Smart Resume Analysis Report", ln = True, align = "C")
    pdf.ln(10)
    
    pdf.cell(200, 10, txt = f"Predicted Job Role: {prediction}", ln = True)
    if match_score is not None:
        pdf.cell(200, 10, txt = f"Resume Match Score: {match_score}%", ln = True)
        
    pdf.ln(5)
    pdf.cell(200, 10, txt = f"Top Keywords: {', '.join(top_words)}", ln = True)
    
    pdf.ln(5)
    pdf.cell(200, 10, txt = f"Technical Skills: {', '.join(tech)}", ln = True)
    pdf.cell(200, 10, txt = f"Soft Skills: {', '.join(soft)}", ln = True)
    
    temp_file = tempfile.NamedTemporaryFile(delete= False, suffix=".pdf")
    pdf.output(temp_file.name)
    return temp_file.name

# StreamLit UI
# StreamLit UI
st.title("Smart Resume Analyser")
st.subheader("Upload your Resume PDF and get your ideal Job Role")

uploaded_file = st.file_uploader("Choose PDF Resume file", type=[".pdf"])

if uploaded_file is not None:
    try:
        # Extract resume text
        resume_text = extract_text_from_pdf(uploaded_file)

        if resume_text.strip():
            st.success("Resume text extracted successfully!")
            st.subheader("Extracted Resume Text")
            st.text_area("Text from Resume", resume_text, height=200)

            cleaned_text = clean_text(resume_text)
            vectorized_input = vectorizer.transform([cleaned_text])

            prediction = model.predict(vectorized_input)[0]
            proba = model.predict_proba(vectorized_input)

            # Confidence Breakdown
            class_names = model.classes_
            confidence = dict(zip(class_names, proba[0]))
            sorted_confidence = dict(sorted(confidence.items(), key=lambda x: x[1], reverse=True))
            st.success(f"Predicted Job Role: {prediction}")

            st.subheader("Prediction Confidence")
            for role, score in list(sorted_confidence.items())[:5]:
                st.write(f"{role}: {round(score * 100, 2)}%")

            st.subheader("Resume with Highlighted Keywords")
            feature_names = vectorizer.get_feature_names_out()
            top_indices = vectorized_input.toarray()[0].argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_indices if vectorized_input.toarray()[0][i] > 0]

            highlighted = highlight_keywords(resume_text, top_words)
            st.markdown(highlighted)

            st.subheader("Detected Skills")
            tech, soft = categorize_skills(cleaned_text)
            st.markdown("**Technical Skills:**")
            st.write(", ".join(tech) if tech else "Not Detected")
            st.markdown("**Soft Skills:**")
            st.write(", ".join(soft) if soft else "Not Detected")

            st.subheader("Match with Job Description")
            jd_text = st.text_area("Paste Job Description here to calculate the match score")

            match_percent = None
            if jd_text:
                jd_cleaned = clean_text(jd_text)
                jd_vec = vectorizer.transform([jd_cleaned])
                similarity = np.dot(vectorized_input.toarray(), jd_vec.toarray().T)[0][0]
                match_percent = round(similarity * 100, 2)
                st.success(f"Resume Score: {match_percent}%")

            if st.button("Export PDF Report"):
                report_path = generate_pdf_report(prediction, top_words, tech, soft, match_score=match_percent)
                with open(report_path, "rb") as f:
                    st.download_button("Download Resume Analysis Report", f, file_name="resume_analysis_report.pdf")
    except Exception as e:
        st.error(f"An error occurred: {e}")
