import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
import PyPDF2
import requests
import json
import re
import string
import io
from collections import Counter
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Mentor",
    page_icon="🤖",
    layout="wide"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }
    h1 { color: #00d4ff; font-size: 2.5rem; }
    h2, h3 { color: #e0e0e0; }
    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #0077b6);
        color: white; border: none; border-radius: 8px;
        padding: 0.6rem 2rem; font-weight: bold;
    }
    .skill-tag {
        display: inline-block;
        background: #1e3a5f; color: #00d4ff;
        border-radius: 20px; padding: 3px 12px;
        margin: 3px; font-size: 0.85rem;
    }
    .missing-tag {
        display: inline-block;
        background: #3a1e1e; color: #ff6b6b;
        border-radius: 20px; padding: 3px 12px;
        margin: 3px; font-size: 0.85rem;
    }
    .job-card {
        background: #1a1d2e; border: 1px solid #2a2d3e;
        border-radius: 10px; padding: 1rem; margin: 0.5rem 0;
    }
    .career-card {
        background: #1a1d2e; border-left: 4px solid #00d4ff;
        border-radius: 8px; padding: 1rem; margin: 0.5rem 0;
    }
    .section-header {
        border-bottom: 1px solid #2a2d3e;
        padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
MULTI_WORD_SKILLS = [
    "machine learning", "deep learning", "natural language processing", "nlp",
    "data science", "web development", "full stack", "front end", "back end",
    "cloud computing", "amazon web services", "aws", "google cloud platform", "gcp",
    "microsoft azure", "scikit-learn", "data analysis", "big data", "agile methodology",
    "project management", "computer vision", "artificial intelligence", "ai", "mern stack",
    "rag pipelines", "fastapi", "streamlit", "git hub", "computer networks",
    "workflow automation", "discrete structures", "probability & statistics",
    "object oriented programming", "oop", "data structures and algorithms", "dsa"
]

# ── Helper functions ──────────────────────────────────────────────────────────
def to_lowercase(text): return text.lower()
def remove_punctuation(text): return text.translate(str.maketrans('', '', string.punctuation))
def remove_brackets(text): return re.sub(r'\[.*?\]', '', text)

def advanced_tokenize_skills(skills_text):
    if pd.isna(skills_text) or not isinstance(skills_text, str):
        return []
    skills_text_lower = skills_text.lower()
    skill_terms = []
    for mw_skill in sorted(MULTI_WORD_SKILLS, key=len, reverse=True):
        if mw_skill in skills_text_lower:
            skill_terms.append(mw_skill)
            skills_text_lower = skills_text_lower.replace(mw_skill, " ", 1)
    remaining_parts = re.split(r'[,;]|\sand\s', skills_text_lower)
    for part in remaining_parts:
        part = part.strip()
        if not part:
            continue
        for term in re.split(r'\s+|[^a-z0-9-]', part):
            term = term.strip()
            if term and len(term) > 1 and not term.isdigit() and term not in [
                'eg', 'etc', 'skills', 'and', 'or', 'a', 'the', 'of', 'in', 'for', 'with', 'on', 'to', 'from'
            ]:
                skill_terms.append(term)
    return list(set(skill_terms))

def extract_text_from_pdf(uploaded_file):
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() or ''
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def extract_json_skills(text):
    text = re.sub(r'```json|```', '', text).strip()
    start = text.rfind('[')
    end = text.rfind(']')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    return []

def get_live_jobs(job_title, location, api_key):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {
        "X-RapidAPI-Key": api_key,
        "X-RapidAPI-Host": "jsearch.p.rapidapi.com"
    }
    params = {"query": f"{job_title} in {location}"}
    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        st.warning(f"Job search error: {e}")
        return []

def download_dataset_from_kaggle():
    """Download job_descriptions.csv from Kaggle if not present locally."""
    if os.path.exists('job_descriptions.csv'):
        return True
    try:
        import kaggle
        kaggle.api.authenticate()
        with st.spinner("📥 Downloading job descriptions dataset from Kaggle (one-time, ~1.6GB)..."):
            kaggle.api.dataset_download_files(
                'bcsf24m006asadullah/job-desciptions',
                path='.',
                unzip=True
            )
        if os.path.exists('job_descriptions.csv'):
            st.success("✅ Dataset downloaded successfully!")
            return True
        else:
            st.warning("⚠️ Download completed but file not found. Proceeding without it.")
            return False
    except Exception as e:
        st.warning(f"⚠️ Could not download from Kaggle: {e}. Proceeding with job_dataset.csv only.")
        return False

# ── Cached loaders ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Building career index...")
def load_data_and_index():

    # ── Download from Kaggle if missing ──────────────────────────────────────
    download_dataset_from_kaggle()

    # ── Load job_descriptions.csv ─────────────────────────────────────────────
    try:
        df1 = pd.read_csv('job_descriptions.csv', engine='python', on_bad_lines='skip')
        job_skills_df = df1[['Job Title', 'skills']].dropna().copy()
    except Exception as e:
        st.warning(f"Could not load job_descriptions.csv: {e}")
        job_skills_df = pd.DataFrame(columns=['Job Title', 'skills'])

    # ── Load job_dataset.csv ──────────────────────────────────────────────────
    try:
        df2 = pd.read_csv('job_dataset.csv', engine='python', on_bad_lines='skip')
        df2_selected = df2[['Title', 'Keywords']].copy()
        df2_selected.rename(columns={'Title': 'Job Title', 'Keywords': 'skills'}, inplace=True)
    except Exception as e:
        st.warning(f"Could not load job_dataset.csv: {e}")
        df2_selected = pd.DataFrame(columns=['Job Title', 'skills'])

    combined = pd.concat([job_skills_df, df2_selected], ignore_index=True).dropna()

    # ── Preprocess ────────────────────────────────────────────────────────────
    for col in ['Job Title', 'skills']:
        combined[col] = combined[col].apply(to_lowercase)
        combined[col] = combined[col].apply(remove_punctuation)
        combined[col] = combined[col].apply(remove_brackets)

    # ── Aggregate skills by job title ─────────────────────────────────────────
    job_title_to_skills = (
        combined.groupby('Job Title')['skills']
        .apply(lambda x: ' '.join(x.dropna().unique()))
        .reset_index()
    )
    unique_titles = job_title_to_skills['Job Title'].tolist()
    combined_skills = job_title_to_skills['skills'].tolist()

    # ── Build job skills set ──────────────────────────────────────────────────
    all_skills = [s for sublist in combined['skills'].apply(advanced_tokenize_skills) for s in sublist]
    job_skills_set = set(all_skills)

    # ── Build FAISS index ─────────────────────────────────────────────────────
    embedder = load_embedder()
    embeddings = embedder.encode(combined_skills, convert_to_tensor=False).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return combined, job_title_to_skills, unique_titles, job_skills_set, index

# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("<h1>🤖 AI Career Mentor</h1>", unsafe_allow_html=True)
st.markdown("Upload your resume and get **personalized career recommendations**, skill gap analysis, a learning roadmap, and live job opportunities.")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    groq_api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    rapidapi_key = st.text_input("RapidAPI Key (JSearch)", type="password", placeholder="your_key...")
    location = st.text_input("Job Search Location", value="Lahore, Pakistan")
    top_k = st.slider("Number of Career Recommendations", 3, 10, 5)

    st.markdown("---")
    st.markdown("### 📦 Kaggle Dataset Setup")
    st.markdown("""
To auto-download the dataset, place your `kaggle.json` in:
- **Linux/Mac:** `~/.kaggle/kaggle.json`
- **Windows:** `C:/Users/<username>/.kaggle/kaggle.json`

Get it from: [kaggle.com](https://www.kaggle.com) → Account → API → Create Token
""")
    st.markdown("---")
    st.markdown("**Built by** [Asadullah Javed](https://asadullahjvd.github.io)")

# ── Main content ──────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload PDF resume", type="pdf")

with col2:
    st.subheader("💡 Additional Info")
    user_interests = st.text_area("Your Interests", placeholder="e.g. AI, Machine Learning, Web Development...")

st.markdown("---")

if st.button("🚀 Get Career Recommendations", use_container_width=True):

    # ── Validation ────────────────────────────────────────────────────────────
    if not uploaded_file:
        st.error("Please upload your resume PDF.")
        st.stop()
    if not groq_api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        st.stop()

    # ── Step 1: Load data & model ─────────────────────────────────────────────
    with st.spinner("Loading career data..."):
        combined, job_title_to_skills, unique_titles, job_skills_set, faiss_index = load_data_and_index()
        embedder = load_embedder()

    # ── Step 2: Extract resume text ───────────────────────────────────────────
    with st.spinner("Reading your resume..."):
        resume_text = extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
        if not resume_text:
            st.error("Could not extract text from PDF.")
            st.stop()

    # ── Step 3: Extract skills via LLM ───────────────────────────────────────
    with st.spinner("Extracting skills from resume..."):
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
        prompt = f"""
You are a skill extraction system. Extract ALL technical and soft skills from the resume below.
Return ONLY a valid JSON array of skill names. No explanation, no markdown, no preamble.
Example: ["Python", "Machine Learning", "SQL", "Communication"]

Resume:
{resume_text}
"""
        response = llm.invoke(prompt)
        extracted_skills = extract_json_skills(response.content)
        if not extracted_skills:
            st.error("Could not extract skills from resume. Try again.")
            st.stop()
        extracted_skills_lower = [s.lower() for s in extracted_skills]

    # ── Display extracted skills ──────────────────────────────────────────────
    st.markdown("<div class='section-header'><h3>🛠️ Skills Extracted from Your Resume</h3></div>", unsafe_allow_html=True)
    skills_html = "".join([f"<span class='skill-tag'>{s}</span>" for s in extracted_skills])
    st.markdown(skills_html, unsafe_allow_html=True)

    # ── Step 4: FAISS recommendation ─────────────────────────────────────────
    with st.spinner("Finding best career matches..."):
        combined_skills_str = ", ".join(extracted_skills_lower)
        cv_embedding = embedder.encode(combined_skills_str, convert_to_tensor=False).astype('float32')
        distances, indices = faiss_index.search(np.array([cv_embedding]), top_k)

    # ── Display career recommendations ───────────────────────────────────────
    st.markdown("<div class='section-header'><h3>🎯 Recommended Career Paths</h3></div>", unsafe_allow_html=True)

    recommended_careers = []
    user_skills_set = set(extracted_skills_lower)

    for i in range(top_k):
        idx = indices[0][i]
        career_title = unique_titles[idx]
        similarity = round(1 - (distances[0][i] / 2), 4)

        skills_str = job_title_to_skills[job_title_to_skills['Job Title'] == career_title]['skills'].iloc[0]
        required_skills = set(advanced_tokenize_skills(skills_str))
        missing_skills = sorted(required_skills - user_skills_set)

        recommended_careers.append({
            "title": career_title.title(),
            "similarity": similarity,
            "missing_skills": missing_skills
        })

        with st.expander(f"{i+1}. {career_title.title()} — Match: {similarity:.2%}"):
            if missing_skills:
                st.markdown("**❌ Skills you need to develop:**")
                missing_html = "".join([f"<span class='missing-tag'>{s}</span>" for s in missing_skills[:15]])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.success("You have all major skills for this role!")

    # ── Step 5: User selects career ───────────────────────────────────────────
    st.markdown("<div class='section-header'><h3>🗺️ Learning Roadmap</h3></div>", unsafe_allow_html=True)
    career_options = [c['title'] for c in recommended_careers]
    selected_career = st.selectbox("Select a career path to get your roadmap:", career_options)

    if selected_career:
        selected_info = next(c for c in recommended_careers if c['title'] == selected_career)
        missing = selected_info['missing_skills']

        # ── Step 6: LLM Roadmap ───────────────────────────────────────────────
        with st.spinner("Generating your personalized roadmap..."):
            roadmap_prompt = f"""
You are an expert career advisor. The user wants to become a {selected_career}.

Their current skills: {', '.join(extracted_skills_lower[:20])}
Their interests: {user_interests}
Missing skills for this role: {', '.join(missing[:15]) if missing else 'None — they are well prepared!'}

Generate a clear, structured learning roadmap with:
1. Phase-by-phase plan (Phase 1, Phase 2, Phase 3)
2. Specific resources/courses for each missing skill
3. Estimated timeline per phase
4. A final tip for landing their first job/internship

Format as clean markdown.
"""
            roadmap_response = llm.invoke(roadmap_prompt)
            st.markdown(roadmap_response.content)

        # ── Step 7: Live Jobs ─────────────────────────────────────────────────
        st.markdown("<div class='section-header'><h3>💼 Live Job Opportunities</h3></div>", unsafe_allow_html=True)

        if rapidapi_key:
            with st.spinner(f"Searching live jobs for {selected_career}..."):
                jobs = get_live_jobs(selected_career, location, rapidapi_key)

            if jobs:
                for job in jobs[:8]:
                    title = job.get("job_title", "N/A")
                    company = job.get("employer_name", "N/A")
                    job_loc = job.get("job_city", location)
                    link = job.get("job_apply_link", "#")
                    employment_type = job.get("job_employment_type", "")
                    st.markdown(f"""
<div class='job-card'>
    <strong>{title}</strong><br>
    🏢 {company} &nbsp;|&nbsp; 📍 {job_loc} &nbsp;|&nbsp; 📌 {employment_type}<br>
    <a href='{link}' target='_blank'>👉 Apply Now</a>
</div>
""", unsafe_allow_html=True)
            else:
                st.info("No live jobs found for this role in your location. Try a different location.")
        else:
            st.warning("Enter your RapidAPI key in the sidebar to see live job listings.")

        # ── Step 8: LLM Job Search Guidance ──────────────────────────────────
        with st.spinner("Generating job search guidance..."):
            guidance_prompt = f"""
You are a job search advisor. The user is a fresh CS graduate targeting: {selected_career} in {location}.

Provide actionable job search guidance in clean markdown:
1. **Top Search Queries** — 3-5 queries to paste into LinkedIn/Indeed
2. **Best Platforms** — where to find these jobs in Pakistan/remotely
3. **Key Companies** — who hires for this role in Pakistan's tech industry
4. **Quick Tips** — 3 tips to stand out as a fresher

Keep it concise and practical.
"""
            guidance_response = llm.invoke(guidance_prompt)
            st.markdown("#### 🔍 Job Search Guidance")
            st.markdown(guidance_response.content)
