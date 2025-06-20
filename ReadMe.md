# 💼 Smart Resume Analyzer

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://kanishk2254-smart-resume-analyser.streamlit.app)

A machine learning–powered web app to analyze resumes, extract keywords, detect skills, and recommend the most suitable job role — all in one click!

---

## 🚀 Live Demo

🔗 **Try it now** → [Smart Resume Analyzer](https://kanishk2254-smart-resume-analyser.streamlit.app)

---

## 🧠 Features

- 📄 Upload your resume in **PDF format**
- 🧹 **Clean and process** resume text automatically
- 🤖 Predict the **best-fit job role** using ML (TF-IDF + classification)
- 📊 Show **confidence levels** for top 5 job categories
- 🧠 Highlight **important keywords** detected
- 🛠️ Categorize detected **technical** and **soft skills**
- 📎 Compare resume with **job description** and get a **match score**
- 📤 Download a full **PDF analysis report**

---

## 🛠️ Tech Stack

| Layer      | Tools Used                              |
|------------|------------------------------------------|
| 💻 Frontend | [Streamlit](https://streamlit.io)       |
| 🧠 ML Model | TF-IDF + Logistic Regression (sklearn)  |
| 📦 Backend  | Python, joblib, PyPDF2, fpdf             |
| 📁 Data     | [Kaggle Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset) |

---

## 📝 How to Run Locally

1. **Clone the repo:**
   git clone https://github.com/Kanishk2254/smart-resume-analyser.git
   cd smart-resume-analyser
2. **Install Dependencies:**
    pip install -r requirements.txt
3. **Run the APP:**
   streamlit run Resume_Analyser_APP_UI.py


## 📁 Project Structure
📦 smart-resume-analyser
├── Resume_Analyser_APP_UI.py          # Streamlit app
├── train_job_role_classifier.py       # ML model training script
├── job_model.pkl                      # Trained classifier
├── vectorizer.pkl                     # TF-IDF vectorizer
├── Resume.csv                         # Resume dataset
├── requirements.txt                   # Required Python packages
└── README.md                          # This file

## 📌ScreenShots
![image](https://github.com/user-attachments/assets/efab890f-81ff-4470-bf9d-c7ca63c38f92)


## 🙌 Credits

• Resume Dataset by Snehaan Bhawal on Kaggle
• Built using open-source tools: Streamlit, scikit-learn, PyPDF2, fpdf

## 📩 Contact
🔗 www.linkedin.com/in/kanishk-mahajan-95009b303
📧 kanishkm445@gmail.com 
