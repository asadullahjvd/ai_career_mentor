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

# ── Custom CSS with Animations ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    * { font-family: 'Inter', sans-serif; }
    .main { background-color: #0f1117; }
    .block-container { padding-top: 2rem; }

    h1 {
        color: #00d4ff;
        font-size: 2.5rem;
        animation: fadeSlideDown 0.8s ease forwards;
    }
    h2, h3 { color: #e0e0e0; }

    .stButton>button {
        background: linear-gradient(135deg, #00d4ff, #0077b6, #00d4ff);
        background-size: 200% auto;
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.7rem 2rem;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.4s ease;
        animation: pulse 2s infinite;
    }
    .stButton>button:hover {
        background-position: right center;
        transform: scale(1.03);
        box-shadow: 0 0 20px rgba(0, 212, 255, 0.5);
    }

    .skill-tag {
        display: inline-block;
        background: #1e3a5f;
        color: #00d4ff;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.85rem;
        animation: popIn 0.4s ease forwards;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .skill-tag:hover {
        transform: scale(1.1);
        box-shadow: 0 0 10px rgba(0, 212, 255, 0.4);
    }

    .missing-tag {
        display: inline-block;
        background: #3a1e1e;
        color: #ff6b6b;
        border-radius: 20px;
        padding: 4px 14px;
        margin: 4px;
        font-size: 0.85rem;
        animation: popIn 0.4s ease forwards;
        transition: transform 0.2s ease;
    }
    .missing-tag:hover { transform: scale(1.1); }

    .job-card {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin: 0.6rem 0;
        animation: slideInLeft 0.5s ease forwards;
        transition: border-color 0.3s ease, box-shadow 0.3s ease;
    }
    .job-card:hover {
        border-color: #00d4ff;
        box-shadow: 0 0 15px rgba(0, 212, 255, 0.2);
    }
    .job-card a { color: #00d4ff; text-decoration: none; font-weight: 600; }
    .job-card a:hover { text-decoration: underline; }

    .career-card {
        background: #1a1d2e;
        border-left: 4px solid #00d4ff;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        animation: slideInLeft 0.5s ease forwards;
    }

    .section-header {
        border-bottom: 1px solid #2a2d3e;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        animation: fadeIn 0.6s ease forwards;
    }

    .hero-banner {
        background: linear-gradient(135deg, #0d1b2a, #1a1d2e, #0d2137);
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 2rem;
        animation: fadeSlideDown 0.8s ease forwards;
        text-align: center;
    }
    .hero-banner h2 {
        color: #00d4ff;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .hero-banner p {
        color: #a0aec0;
        font-size: 1rem;
        margin: 0;
    }

    .stat-card {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        animation: popIn 0.6s ease forwards;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .stat-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.15);
    }
    .stat-card .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4ff;
    }
    .stat-card .stat-label {
        font-size: 0.85rem;
        color: #a0aec0;
        margin-top: 4px;
    }

    .step-card {
        background: #1a1d2e;
        border-left: 3px solid #00d4ff;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        animation: slideInLeft 0.5s ease forwards;
    }
    .step-number {
        background: #00d4ff;
        color: #0f1117;
        border-radius: 50%;
        width: 28px;
        height: 28px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 10px;
    }

    /* Keyframe animations */
    @keyframes fadeSlideDown {
        from { opacity: 0; transform: translateY(-20px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeIn {
        from { opacity: 0; }
        to   { opacity: 1; }
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.8); }
        to   { opacity: 1; transform: scale(1); }
    }
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to   { opacity: 1; transform: translateX(0); }
    }
    @keyframes pulse {
        0%   { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0.4); }
        70%  { box-shadow: 0 0 0 10px rgba(0, 212, 255, 0); }
        100% { box-shadow: 0 0 0 0 rgba(0, 212, 255, 0); }
    }
    @keyframes shimmer {
        0%   { background-position: -200% center; }
        100% { background-position: 200% center; }
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: #0d1b2a;
        border-right: 1px solid #1e3a5f;
    }
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #00d4ff;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #1e3a5f;
        border-radius: 12px;
        transition: border-color 0.3s ease;
    }
    [data-testid="stFileUploader"]:hover {
        border-color: #00d4ff;
    }

    /* Expander styling */
    [data-testid="stExpander"] {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        transition: border-color 0.3s ease;
    }
    [data-testid="stExpander"]:hover {
        border-color: #00d4ff55;
    }

    /* Floating glow effect on main title */
    .glow-title {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d4ff, #0077b6, #00d4ff);
        background-size: 200% auto;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: shimmer 3s linear infinite;
        display: inline-block;
    }

    .subtitle-text {
        color: #a0aec0;
        font-size: 1.1rem;
        animation: fadeIn 1.2s ease forwards;
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

# ── Load secrets safely ───────────────────────────────────────────────────────
def get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except:
        return default

DEFAULT_GROQ_KEY     = get_secret("GROQ_API_KEY")
DEFAULT_RAPIDAPI_KEY = get_secret("RAPIDAPI_KEY")

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
    if os.path.exists('job_descriptions.csv'):
        return True
    try:
        kaggle_dir = os.path.expanduser('~/.kaggle')
        os.makedirs(kaggle_dir, exist_ok=True)
        kaggle_creds = {
            "username": st.secrets["kaggle"]["username"],
            "key": st.secrets["kaggle"]["key"]
        }
        with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
            json.dump(kaggle_creds, f)
        os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)

        import kaggle
        kaggle.api.authenticate()
        with st.spinner("📥 Downloading job descriptions dataset from Kaggle (one-time ~1.6GB)..."):
            kaggle.api.dataset_download_files(
                'bcsf24m006asadullah/job-desciptions',
                path='.', unzip=True
            )
        if os.path.exists('job_descriptions.csv'):
            st.success("✅ Dataset downloaded successfully!")
            return True
        else:
            st.warning("⚠️ Download completed but file not found.")
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
    download_dataset_from_kaggle()

    try:
        df1 = pd.read_csv('job_descriptions.csv', engine='python', on_bad_lines='skip')
        job_skills_df = df1[['Job Title', 'skills']].dropna().copy()
    except Exception as e:
        st.warning(f"Could not load job_descriptions.csv: {e}")
        job_skills_df = pd.DataFrame(columns=['Job Title', 'skills'])

    try:
        df2 = pd.read_csv('job_dataset.csv', engine='python', on_bad_lines='skip')
        df2_selected = df2[['Title', 'Keywords']].copy()
        df2_selected.rename(columns={'Title': 'Job Title', 'Keywords': 'skills'}, inplace=True)
    except Exception as e:
        st.warning(f"Could not load job_dataset.csv: {e}")
        df2_selected = pd.DataFrame(columns=['Job Title', 'skills'])

    combined = pd.concat([job_skills_df, df2_selected], ignore_index=True).dropna()

    for col in ['Job Title', 'skills']:
        combined[col] = combined[col].apply(to_lowercase)
        combined[col] = combined[col].apply(remove_punctuation)
        combined[col] = combined[col].apply(remove_brackets)

    job_title_to_skills = (
        combined.groupby('Job Title')['skills']
        .apply(lambda x: ' '.join(x.dropna().unique()))
        .reset_index()
    )
    unique_titles = job_title_to_skills['Job Title'].tolist()
    combined_skills = job_title_to_skills['skills'].tolist()

    all_skills = [s for sublist in combined['skills'].apply(advanced_tokenize_skills) for s in sublist]
    job_skills_set = set(all_skills)

    embedder = load_embedder()
    embeddings = embedder.encode(combined_skills, convert_to_tensor=False).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return combined, job_title_to_skills, unique_titles, job_skills_set, index

# ── Hero UI ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero-banner'>
    <div class='glow-title'>🤖 AI Career Mentor</div>
    <p class='subtitle-text'>Upload your resume · Discover career paths · Get your personalized roadmap</p>
</div>
""", unsafe_allow_html=True)

# ── Stat Cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("<div class='stat-card'><div class='stat-number'>10K+</div><div class='stat-label'>Job Titles Indexed</div></div>", unsafe_allow_html=True)
with c2:
    st.markdown("<div class='stat-card'><div class='stat-number'>1.6M</div><div class='stat-label'>Job Descriptions</div></div>", unsafe_allow_html=True)
with c3:
    st.markdown("<div class='stat-card'><div class='stat-number'>AI</div><div class='stat-label'>Powered by Groq LLM</div></div>", unsafe_allow_html=True)
with c4:
    st.markdown("<div class='stat-card'><div class='stat-number'>Live</div><div class='stat-label'>Real-Time Job Search</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("✨ How it works"):
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("<div class='step-card'><span class='step-number'>1</span>Upload your PDF resume</div>", unsafe_allow_html=True)
    with s2:
        st.markdown("<div class='step-card'><span class='step-number'>2</span>AI extracts your skills</div>", unsafe_allow_html=True)
    with s3:
        st.markdown("<div class='step-card'><span class='step-number'>3</span>FAISS matches career paths</div>", unsafe_allow_html=True)
    with s4:
        st.markdown("<div class='step-card'><span class='step-number'>4</span>Get roadmap + live jobs</div>", unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar (no API keys shown) ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    location = st.text_input("📍 Job Search Location", value="Lahore, Pakistan")
    top_k = st.slider("🎯 Career Recommendations", 3, 10, 5)
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#a0aec0; font-size:0.85rem;'>
        🔒 API keys are securely managed<br>by the platform.<br><br>
        No setup required — just upload<br>your resume and go!
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='text-align:center;'>Built by <a href='https://asadullahjvd.github.io' style='color:#00d4ff;'>Asadullah Javed</a></div>", unsafe_allow_html=True)

# ── Use secrets directly (not from sidebar input) ─────────────────────────────
groq_api_key     = DEFAULT_GROQ_KEY
rapidapi_key     = DEFAULT_RAPIDAPI_KEY

# ── Session State ─────────────────────────────────────────────────────────────
if 'clicked_recommend' not in st.session_state:
    st.session_state.clicked_recommend = False
if 'extracted_skills' not in st.session_state:
    st.session_state.extracted_skills = []
if 'extracted_skills_lower' not in st.session_state:
    st.session_state.extracted_skills_lower = []
if 'recommended_careers' not in st.session_state:
    st.session_state.recommended_careers = []

# ── Main Input ────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload PDF resume", type="pdf")

with col2:
    st.subheader("💡 Your Interests")
    user_interests = st.text_area("What excites you?", placeholder="e.g. AI, Machine Learning, Web Development...", height=133)

st.markdown("---")

# ── Button ────────────────────────────────────────────────────────────────────
if st.button("🚀 Get Career Recommendations", use_container_width=True):
    if not uploaded_file:
        st.error("⚠️ Please upload your resume PDF.")
        st.stop()
    if not groq_api_key:
        st.error("⚠️ Groq API key not configured. Please contact the administrator.")
        st.stop()

    try:
        with st.spinner("🔍 Loading career database and building vector index..."):
            combined, job_title_to_skills, unique_titles, job_skills_set, faiss_index = load_data_and_index()
            embedder = load_embedder()

        if len(unique_titles) == 0:
            st.error("Career dataset could not be compiled.")
            st.stop()

        with st.spinner("📖 Reading your resume..."):
            resume_text = extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
            if not resume_text or len(resume_text.strip()) == 0:
                st.error("Could not extract text from PDF. Is it a scanned image?")
                st.stop()

        with st.spinner("🧠 Extracting skills with Groq LLM..."):
            llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
            prompt = f"""
You are a skill extraction system. Extract ALL technical and soft skills from the resume below.
Return ONLY a valid JSON array of skill names. No conversational text, no markdown, no preamble.
Example: ["Python", "Machine Learning", "SQL", "Communication"]

Resume:
{resume_text}
"""
            response = llm.invoke(prompt)
            raw_content = response.content.strip()

            extracted = extract_json_skills(raw_content)
            if not extracted and ("[" in raw_content and "]" in raw_content):
                try:
                    start_idx = raw_content.find('[')
                    end_idx = raw_content.find(']') + 1
                    extracted = json.loads(raw_content[start_idx:end_idx])
                except:
                    pass

            if not extracted:
                cleaned_fallback = re.sub(r'[\[\]\"\'\'\'\"\"]', '', raw_content)
                extracted = [s.strip() for s in cleaned_fallback.split(',') if s.strip()]

            if not extracted:
                st.error("Skill extraction failed. Please try again.")
                st.stop()

            st.session_state.extracted_skills = extracted
            st.session_state.extracted_skills_lower = [s.lower() for s in extracted]

        with st.spinner("⚡ Matching careers with FAISS..."):
            combined_skills_str = ", ".join(st.session_state.extracted_skills_lower)
            cv_embedding = embedder.encode(combined_skills_str, convert_to_tensor=False).astype('float32')
            distances, indices = faiss_index.search(np.array([cv_embedding]), top_k)

        recommended_careers = []
        user_skills_set = set(st.session_state.extracted_skills_lower)

        for i in range(top_k):
            if i >= len(indices[0]):
                break
            idx = indices[0][i]
            if idx >= len(unique_titles):
                continue

            career_title = unique_titles[idx]
            similarity = round(1 - (distances[0][i] / 2), 4)

            skills_match = job_title_to_skills[job_title_to_skills['Job Title'] == career_title]
            if not skills_match.empty:
                skills_str = skills_match['skills'].iloc[0]
                required_skills = set(advanced_tokenize_skills(skills_str))
                missing_skills = sorted(required_skills - user_skills_set)
            else:
                missing_skills = []

            recommended_careers.append({
                "title": career_title.title(),
                "similarity": similarity,
                "missing_skills": missing_skills
            })

        st.session_state.recommended_careers = recommended_careers
        st.session_state.clicked_recommend = True

    except Exception as e:
        st.error(f"Error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.clicked_recommend and st.session_state.extracted_skills:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

    st.markdown("<div class='section-header'><h3>🛠️ Skills Extracted from Your Resume</h3></div>", unsafe_allow_html=True)
    skills_html = "".join([f"<span class='skill-tag'>{s}</span>" for s in st.session_state.extracted_skills])
    st.markdown(skills_html, unsafe_allow_html=True)

    st.markdown("<div class='section-header'><h3>🎯 Recommended Career Paths</h3></div>", unsafe_allow_html=True)
    for i, career in enumerate(st.session_state.recommended_careers):
        with st.expander(f"{i+1}. {career['title']} — Match: {career['similarity']:.2%}"):
            if career['missing_skills']:
                st.markdown("**❌ Skills you need to develop:**")
                missing_html = "".join([f"<span class='missing-tag'>{s}</span>" for s in career['missing_skills'][:15]])
                st.markdown(missing_html, unsafe_allow_html=True)
            else:
                st.success("✅ You have all major skills for this role!")

    st.markdown("<div class='section-header'><h3>🗺️ Learning Roadmap</h3></div>", unsafe_allow_html=True)
    career_options = [c['title'] for c in st.session_state.recommended_careers]

    if career_options:
        selected_career = st.selectbox("Select a career path to get your roadmap:", career_options)

        if selected_career:
            selected_info = next((c for c in st.session_state.recommended_careers if c['title'] == selected_career), None)
            if selected_info:
                missing = selected_info['missing_skills']

                with st.spinner("✍️ Generating your personalized roadmap..."):
                    roadmap_prompt = f"""
You are an expert career advisor. The user wants to become a {selected_career}.

Their current skills: {', '.join(st.session_state.extracted_skills_lower[:20])}
Their interests: {user_interests}
Missing skills for this role: {', '.join(missing[:15]) if missing else 'None — they are well prepared!'}

Generate a clear, structured learning roadmap with:
1. Phase-by-phase plan (Phase 1, Phase 2, Phase 3)
2. Specific resources/courses for each missing skill
3. Estimated timeline per phase
4. A final tip for landing their first job/internship

Format as clean markdown.
"""
                    try:
                        roadmap_response = llm.invoke(roadmap_prompt)
                        st.markdown(roadmap_response.content)
                    except Exception as e:
                        st.error(f"Roadmap generation failed: {e}")

                st.markdown("<div class='section-header'><h3>💼 Live Job Opportunities</h3></div>", unsafe_allow_html=True)

                if rapidapi_key:
                    with st.spinner(f"🔎 Searching live jobs for {selected_career}..."):
                        jobs = get_live_jobs(selected_career, location, rapidapi_key)

                    if jobs:
                        for job in jobs[:8]:
                            title       = job.get("job_title", "N/A")
                            company     = job.get("employer_name", "N/A")
                            job_loc     = job.get("job_city", location)
                            link        = job.get("job_apply_link", "#")
                            emp_type    = job.get("job_employment_type", "")
                            st.markdown(f"""
<div class='job-card'>
    <strong style='color:#e0e0e0; font-size:1rem;'>{title}</strong><br>
    🏢 <span style='color:#a0aec0;'>{company}</span> &nbsp;|&nbsp;
    📍 <span style='color:#a0aec0;'>{job_loc}</span> &nbsp;|&nbsp;
    📌 <span style='color:#a0aec0;'>{emp_type}</span><br><br>
    <a href='{link}' target='_blank'>👉 Apply Now</a>
</div>
""", unsafe_allow_html=True)
                    else:
                        st.info("No live jobs found. Try a different location.")
                else:
                    st.info("Live job search is not configured.")

                with st.spinner("📊 Generating job search guidance..."):
                    guidance_prompt = f"""
You are a job search advisor. The user is a fresh CS graduate targeting: {selected_career} in {location}.

Provide actionable job search guidance in clean markdown:
1. **Top Search Queries** — 3-5 queries to paste into LinkedIn/Indeed
2. **Best Platforms** — where to find these jobs in Pakistan/remotely
3. **Key Companies** — who hires for this role in Pakistan's tech industry
4. **Quick Tips** — 3 tips to stand out as a fresher

Keep it concise and practical.
"""
                    try:
                        guidance_response = llm.invoke(guidance_prompt)
                        st.markdown("#### 🔍 Job Search Guidance")
                        st.markdown(guidance_response.content)
                    except Exception as e:
                        st.error(f"Guidance generation failed: {e}")
