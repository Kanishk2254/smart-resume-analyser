# 📄 Smart Resume Analyzer

An intelligent Streamlit app that predicts the most suitable job role based on a resume PDF, highlights key skills, and provides match scoring against job descriptions. Also exports a full PDF report.

## 🚀 Features

- Upload a resume PDF
- Predict ideal job domain using Machine Learning (TF-IDF + Classifier)
- Highlight top keywords
- Categorize technical and soft skills
- Match score with pasted Job Description
- Export PDF report

## 🛠 Tech Stack

- Python
- Streamlit
- Scikit-learn
- PyPDF2
- FPDF

## 📦 Installation

```bash
pip install -r requirements.txt
streamlit run Resume_Analyser_APP_UI.py
