import streamlit as st
import pdfplumber
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
import tempfile
import os
from collections import defaultdict

# SKILLS
SKILL_CATEGORIES={
    "Programming": ["python", "java", "c++", "css", "html", "javascript"],
    "Data Science":["machine learning", "data analysis", "pandas", "numpy", "scikit-learn"],
    "Web Development":["flask", "django", "react", "nodejs"],
    "Cloud & DevOps":["aws", "azure", "docker", "kubernetes", "git", "jenkins"]
}

# Clean up Resume
def clean_text(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-Z\s]','',text)
    return text

# Extract File
def extract_pdf_text(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

# Generate Report    
def generate_pdf_report(score,missing_keywords):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("arial", size=12)
    pdf.cell(200,10, txt="Smart Resume Analyser Report", ln=1, align="C")
    pdf.multi_cell(0,10, f"Job Match Score : {score}%\n\n Missing Keywords: {','.join(missing_keywords)}")
    temp_dir= tempfile.gettempdir()
    pdf_path= os.path.join(temp_dir, "resume_analysis_report.pdf")
    pdf.output(pdf_path)
    return pdf_path

# Highlight Keywords
def highlight_keywords(text, keywords):
    for word in sorted(keywords, key= len, reverse=True):
        text= re.sub(fr"\\b({re.escape(word)})\\b", r"**\1**", text, flags=re.IGNORECASE)
        return text

# Categorize Skills
def categorize_skills(text):
    skills_found=defaultdict(list)
    text_words=set(text.split())
    for category, skills in SKILL_CATEGORIES.items():
        for skill in skills:
            if skill in text_words:
                skills_found[category].append(skill)
    return skills_found

# STREAMLIT UI
st.set_page_config(page_title="Smart Resume Analyser", layout="centered")
st.title("Smart Resume Analyser")
st.write("Analyse how well your resume matches a job description and identify categorized skills.")

resume_file=st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
jd_file=st.file_uploader("Upload Job Descrition (PDF)",type=["pdf"])

if st.button("Analyse"):
    if resume_file and jd_file:
        with st.spinner("Analyzing......."):
            resume_text=extract_pdf_text(resume_file)
            jd_text=extract_pdf_text(jd_file)
            
            resume_clean=clean_text(resume_text)
            jd_clean=clean_text(jd_text)
            
            vectorizer=TfidfVectorizer()
            vectors=vectorizer.fit_transform([resume_clean,jd_clean])
            score= round(cosine_similarity(vectors[0:1]))
            
            resume_words= set(resume_clean.split())
            jd_words= set(jd_clean.split())
            missing= sorted(jd_words - resume_words)
            
            st.subheader(f"U000F4CA Job Match Score: {score}%")
            
            st.subheader("\U0001F52C Missing Keywords: ")
            st.write(",".join(missing) if missing else "None")
            
            st.subheader("\U0001F4DD Job Descripton With Matched Highlights: ")
            highlighted_jd= highlight_keywords(jd_text, resume_words)
            st.markdown(highlighted_jd)
            
            st.subheader("U0001F52C Categorized Skills Found In Resume: ")
            skill_map= categorize_skills(resume_clean)
            for category, skills in skill_map.items():
                st.markdown(f"**{category}**: {','.join(skills)}")
            
            pdf_path= generate_pdf_report(score, missing)
            with open(pdf_path,"rb") as f:
                st.download_button(label="U0001F4BE Download Report as PDF", data=f, file_name="resume_analysis_report.pdf", mime="application/pdf")
    else:
        st.error("Please upload Resume and Job Description PDFs")
        