import streamlit as st
import torch
import re
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os
import gdown
import zipfile

def download_model_from_drive(file_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, "newmodel.zip")

    # Avoid redownloading if already extracted
    if not os.path.exists(os.path.join(output_dir, "config.json")):
        url = f"https://drive.google.com/uc?id=1046GVAZWgU12mWlehraHaFmxypGMUO-u"
        gdown.download(url, zip_path, quiet=False)

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        os.remove(zip_path)

MODEL_DIR = "newmodel/newmodel"
GOOGLE_DRIVE_FILE_ID = "1046GVAZWgU12mWlehraHaFmxypGMUO-u"

download_model_from_drive(GOOGLE_DRIVE_FILE_ID, MODEL_DIR)

# Force Streamlit to use local .streamlit config
os.environ["XDG_CONFIG_HOME"] = os.path.join(os.getcwd(), ".streamlit")

label_classes = [
    'Testing', 'HR', 'Advocate', 'Arts', 'Sales',
    'Mechanical Engineer', 'Data Science', 'Health and fitness',
    'Civil Engineer', 'Java Developer', 'Business Analyst',
    'SAP Developer', 'Automation Testing', 'Electrical Engineering',
    'Operations Manager', 'Python Developer', 'DevOps Engineer',
    'Network Security Engineer', 'PMO', 'Database', 'Hadoop',
    'ETL Developer', 'DotNet Developer', 'Blockchain', 'Web Designing'
]

# Load model and tokenizer
def load_model_and_tokenizer(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    return model, tokenizer

# Extract text from PDF using PyPDF2
def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def extract_skill_names(text):
    # Normalize entire text to lowercase for case-insensitive matching
    text = text.lower()

    # Whitelist: all lowercase for consistent matching
    skill_whitelist = {
        "python", "sql", "excel", "tableau", "tensorflow", "pandas", "numpy",
        "seaborn", "selenium", "beautiful soup", "scrapy", "flask", "windows",
        "linux", "mac os", "mysql", "aws", "google cloud", "ibm certified data scientist",
        "artificial intelligence", "machine learning", "deep learning", "data analysis",
        "data visualization", "natural language processing", "computer vision",
        "statistical analysis", "web scraping", "data mining", "report writing",
        "cloud computing", "docker", "kubernetes", "git", "javascript", "node.js",
        "react", "vue.js", "java", "c++", "c#", "php", "html", "css", "rest api",
        "graphql", "agile", "scrum", "devops", "bash", "powershell",
        "android studio", "genymotion", "android sdk", "android development tools (adt)",
        "json", "xml", "financial modeling", "financial reporting", "financial accounting",
        "business valuation", "sas", "budgeting", "investment analysis", "financial planning",
        "risk management", "invision", "axure",
        "observation", "decision making", "communication", "multi-tasking", "teamwork",
        "problem solving", "wordpress", "sass", "letterpress printing", "graphic design",
        "art direction", "web design", "branding", "mockups", "photo editing", "video editing"
    }

    # Find skill-like phrases in text, split by commas, semicolons, newlines, slashes
    skill_candidates = re.split(r'[,;/\n]+', text)

    extracted_skills = set()

    for candidate in skill_candidates:
        skill = candidate.strip()
        # Clean trailing punctuation like dots or dashes
        skill = re.sub(r'[\.-]+$', '', skill)

        # Try direct whitelist match or substring match to whitelist
        if skill in skill_whitelist:
            extracted_skills.add(skill)
        else:
            # Check if skill contains a whitelist skill as substring (handles minor variations)
            for wskill in skill_whitelist:
                if wskill in skill:
                    extracted_skills.add(wskill)

    # Capitalize skills for better formatting
    def capitalize_skill(s):
        return ' '.join(word.capitalize() for word in s.split())

    # Return sorted list for consistency
    return ', '.join(capitalize_skill(s) for s in sorted(extracted_skills))



# Predict job category and extract skills
def predict_and_extract_skills(resume_text, model, tokenizer):
    inputs = tokenizer(resume_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    predicted_label = label_classes[predicted_class_id]
    extracted_skills = extract_skill_names(resume_text)
    return predicted_label, extracted_skills

# Generate interview questions based on category and skills
def generate_interview_questions(category, skills):
    questions_map = {
        'Data Science': [
            "Explain the difference between supervised and unsupervised learning.",
            "How do you handle missing data in a dataset?",
            "What is overfitting and how can you prevent it?"
        ],
        'HR': [
            "How do you handle conflict resolution in the workplace?",
            "What strategies do you use for talent acquisition?",
            "How do you ensure compliance with labor laws?"
        ],
        'Advocate': [
            "What experience do you have with litigation or client representation?",
            "How do you prepare for court cases?",
            "Can you explain the importance of client confidentiality?"
        ],
        'Arts': [
            "What inspires your creative process?",
            "How do you handle criticism of your work?",
            "Describe a challenging project and how you completed it."
        ],
        'Web Designing': [
            "What design tools and software are you proficient with?",
            "Explain the importance of responsive design.",
            "How do you optimize a website for performance?"
        ],
        'Mechanical Engineer': [
            "Describe a project where you applied thermodynamics.",
            "How do you approach failure analysis in mechanical systems?",
            "What CAD software are you familiar with?"
        ],
        'Sales': [
            "How do you handle rejection during sales calls?",
            "Describe a time you exceeded your sales targets.",
            "What techniques do you use to identify customer needs?"
        ],
        'Health and fitness': [
            "How do you customize fitness plans for clients with different needs?",
            "What certifications do you hold?",
            "Explain how you track and measure client progress."
        ],
        'Civil Engineer': [
            "Describe your experience with project management in construction.",
            "How do you ensure compliance with safety regulations?",
            "What software do you use for structural analysis?"
        ],
        'Java Developer': [
            "Explain the concept of OOP in Java.",
            "How do you handle exceptions in Java?",
            "What experience do you have with Java frameworks like Spring?"
        ],
        'Business Analyst': [
            "How do you gather requirements from stakeholders?",
            "Describe a challenging project and how you managed scope changes.",
            "What tools do you use for process modeling?"
        ],
        'SAP Developer': [
            "Explain your experience with SAP modules.",
            "How do you handle custom development in SAP?",
            "What debugging tools do you use in SAP?"
        ],
        'Automation Testing': [
            "What test automation tools have you worked with?",
            "How do you design a test automation framework?",
            "Explain the difference between manual and automated testing."
        ],
        'Electrical Engineering': [
            "Describe your experience with circuit design.",
            "How do you troubleshoot electrical systems?",
            "What simulation tools do you use?"
        ],
        'Operations Manager': [
            "How do you manage cross-functional teams?",
            "Explain how you optimize operational efficiency.",
            "Describe a time when you resolved a major operational issue."
        ],
        'Python Developer': [
            "What Python libraries do you use for data manipulation?",
            "How do you manage package dependencies?",
            "Explain Python's GIL and its implications."
        ],
        'DevOps Engineer': [
            "What CI/CD tools have you implemented?",
            "How do you ensure infrastructure as code?",
            "Describe your experience with containerization."
        ],
        'Network Security Engineer': [
            "How do you secure a corporate network?",
            "Explain the differences between IDS and IPS.",
            "What experience do you have with firewalls and VPNs?"
        ],
        'PMO': [
            "How do you track project progress?",
            "Describe your experience with risk management.",
            "What tools do you use for project portfolio management?"
        ],
        'Database': [
            "Explain normalization and denormalization.",
            "How do you optimize SQL queries?",
            "Describe your experience with NoSQL databases."
        ],
        'Hadoop': [
            "What components make up the Hadoop ecosystem?",
            "How do you handle large-scale data processing?",
            "Explain MapReduce in your own words."
        ],
        'ETL Developer': [
            "Describe your experience designing ETL workflows.",
            "How do you handle data quality issues?",
            "What ETL tools have you worked with?"
        ],
        'DotNet Developer': [
            "What is the .NET framework?",
            "Explain how you manage memory in .NET applications.",
            "Describe your experience with ASP.NET MVC."
        ],
        'Blockchain': [
            "What is a smart contract?",
            "Explain the concept of decentralization.",
            "Describe a blockchain project you've worked on."
        ],
        'Testing': [
            "What types of testing have you performed?",
            "How do you write effective test cases?",
            "Explain the bug lifecycle."
        ],
    }



    skill_questions_map = {
        'python': "Can you describe your experience with Python libraries such as Pandas or NumPy?",
        'java': "What is your experience with Java frameworks like Spring or Hibernate?",
        'data science': "How do you validate the models you build in data science projects?",
        'machine learning': "Explain how you would select features for a machine learning model.",
        'devops': "What CI/CD tools have you used and how did you implement pipelines?",
        'sql': "Can you write a query to find duplicate records in a SQL table?",
    }

    questions = questions_map.get(category, ["Tell me about yourself and your experience relevant to this role."])

    skill_lower = skills.lower()
    for skill_key, question in skill_questions_map.items():
        if skill_key in skill_lower:
            questions.append(question)

    return questions


# Load model and tokenizer at the start
try:
    # model, tokenizer = load_model_and_tokenizer("D:\Resume-Screening-App-main-main\ResumeAnalyzer\src")
    model, tokenizer = load_model_and_tokenizer(MODEL_DIR)
except Exception as e:
    st.error(f"Failed to load model/tokenizer: {e}")
    st.stop()

# Streamlit UI
st.title("Resume Job Category Prediction, Skill Extraction & Interview Questions")

upload_option = st.radio("Choose input method:", ["Upload PDF", "Paste Text"])
resume_text = ""

if upload_option == "Upload PDF":
    pdf_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
    if pdf_file:
        resume_text = extract_text_from_pdf(pdf_file)
        st.text_area("Extracted Resume Text", value=resume_text, height=300)
else:
    resume_text = st.text_area("Paste the resume text here:")


if st.button("Predict & Generate Interview Questions"):
    if resume_text.strip():
        try:
            with st.spinner('Predicting...'):
                category, skills = predict_and_extract_skills(resume_text, model, tokenizer)
            st.success(f"**Predicted Job Category:** {category}")
            st.info(f"**Extracted Skills:** {skills if skills else 'No skills detected'}")

            interview_questions = generate_interview_questions(category, skills)
            st.subheader("Suggested Interview Questions")
            for i, q in enumerate(interview_questions, 1):
                st.write(f"{i}. {q}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Please provide resume input either by uploading or pasting text.")
