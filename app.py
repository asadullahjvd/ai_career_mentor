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
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Career Mentor",
    page_icon="🤖",
    layout="wide"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }
.main { background-color: #060810; }
.block-container { padding-top: 2rem; }
h2, h3 { color: #e0e0e0; }

.stButton>button {
    background: linear-gradient(135deg, #00d4ff, #0055aa, #7b2fff);
    background-size: 300% auto;
    color: white; border: none; border-radius: 14px;
    padding: 0.8rem 2rem; font-weight: 700; font-size: 1.05rem;
    letter-spacing: 0.5px;
    transition: all 0.4s ease;
    animation: gradientShift 4s ease infinite, glowPulse 2.5s ease-in-out infinite;
    position: relative; overflow: hidden;
}
.stButton>button:hover {
    transform: translateY(-3px) scale(1.02);
    box-shadow: 0 0 30px rgba(0,212,255,0.6), 0 0 60px rgba(0,212,255,0.2);
}

.skill-tag {
    display: inline-block;
    background: linear-gradient(135deg, #0d2137, #0a1628);
    color: #00d4ff; border: 1px solid #1e3a5f;
    border-radius: 20px; padding: 5px 14px;
    margin: 4px; font-size: 0.82rem; font-weight: 500;
    animation: popIn 0.5s cubic-bezier(0.175,0.885,0.32,1.275) forwards;
    transition: all 0.25s ease; cursor: default;
}
.skill-tag:hover {
    background: linear-gradient(135deg, #1e3a5f, #0d2137);
    border-color: #00d4ff;
    transform: translateY(-3px) scale(1.08);
    box-shadow: 0 4px 15px rgba(0,212,255,0.3);
}

.missing-tag {
    display: inline-block;
    background: linear-gradient(135deg, #2a0d0d, #1a0808);
    color: #ff6b6b; border: 1px solid #5f1e1e;
    border-radius: 20px; padding: 5px 14px;
    margin: 4px; font-size: 0.82rem; font-weight: 500;
    animation: popIn 0.5s cubic-bezier(0.175,0.885,0.32,1.275) forwards;
    transition: all 0.25s ease;
}
.missing-tag:hover {
    transform: translateY(-3px) scale(1.08);
    box-shadow: 0 4px 15px rgba(255,107,107,0.3);
}

.job-card {
    background: linear-gradient(135deg, #0d1117, #111827);
    border: 1px solid #1e2a3a; border-radius: 14px;
    padding: 1.2rem 1.4rem; margin: 0.7rem 0;
    animation: slideInLeft 0.5s ease forwards;
    transition: all 0.3s ease;
    position: relative; overflow: hidden;
}
.job-card::before {
    content: '';
    position: absolute; left: 0; top: 0;
    width: 3px; height: 100%;
    background: linear-gradient(180deg, #00d4ff, #7b2fff);
    border-radius: 3px 0 0 3px;
}
.job-card:hover {
    border-color: #00d4ff44;
    transform: translateX(6px);
    box-shadow: 0 8px 30px rgba(0,212,255,0.1), -4px 0 20px rgba(0,212,255,0.15);
}
.job-card a { color: #00d4ff; text-decoration: none; font-weight: 600; transition: color 0.2s ease; }
.job-card a:hover { color: #7b2fff; text-decoration: underline; }

.section-header {
    border-bottom: 1px solid #1e2a3a;
    padding-bottom: 0.6rem; margin: 2rem 0 1rem 0;
    animation: fadeIn 0.6s ease forwards;
}
.section-header h3 {
    background: linear-gradient(90deg, #00d4ff, #7b2fff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 1.4rem; font-weight: 700;
}

[data-testid="stExpander"] {
    background: linear-gradient(135deg, #0d1117, #111827);
    border: 1px solid #1e2a3a; border-radius: 12px;
    margin-bottom: 0.6rem; transition: all 0.3s ease; overflow: hidden;
}
[data-testid="stExpander"]:hover {
    border-color: #00d4ff33;
    box-shadow: 0 4px 20px rgba(0,212,255,0.08);
}

[data-testid="stFileUploader"] {
    border: 2px dashed #1e2a3a; border-radius: 14px;
    background: #0d1117; transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: #00d4ff55;
    box-shadow: 0 0 20px rgba(0,212,255,0.05);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060810, #0d1117);
    border-right: 1px solid #1e2a3a;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #00d4ff; }

.step-card {
    background: linear-gradient(135deg, #0d1117, #111827);
    border: 1px solid #1e2a3a; border-radius: 12px;
    padding: 1.2rem; margin: 0.4rem 0;
    text-align: center; transition: all 0.3s ease;
    animation: popIn 0.6s ease forwards;
}
.step-card:hover {
    border-color: #00d4ff55;
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0,212,255,0.1);
}
.step-icon { font-size: 2rem; margin-bottom: 0.5rem; }
.step-title { color: #00d4ff; font-weight: 600; font-size: 0.95rem; margin-bottom: 0.3rem; }
.step-desc { color: #6b7280; font-size: 0.8rem; }

.hero-wrapper {
    position: relative; text-align: center;
    padding: 3rem 2rem; margin-bottom: 1rem;
    background: linear-gradient(135deg, #060810, #0d1117, #060d1a);
    border: 1px solid #1e2a3a; border-radius: 20px;
    overflow: hidden;
    animation: fadeIn 1s ease forwards;
}
.hero-wrapper::before {
    content: '';
    position: absolute; top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at center, rgba(0,212,255,0.04) 0%, transparent 60%);
    animation: rotateBg 20s linear infinite;
}
.hero-wrapper::after {
    content: '';
    position: absolute; top: 0; left: 0;
    width: 100%; height: 1px;
    background: linear-gradient(90deg, transparent, #00d4ff44, #7b2fff44, transparent);
}

.glow-title {
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(90deg, #00d4ff, #7b2fff, #00d4ff);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    display: block; margin-bottom: 1rem;
    filter: drop-shadow(0 0 20px rgba(0,212,255,0.3));
}

.subtitle-text {
    color: #6b7280; font-size: 1.05rem;
    animation: fadeIn 1.5s ease forwards;
    line-height: 1.6;
}

.badge {
    display: inline-block;
    background: linear-gradient(135deg, #0d2137, #0a1628);
    color: #00d4ff; border: 1px solid #1e3a5f;
    border-radius: 20px; padding: 4px 14px;
    font-size: 0.78rem; font-weight: 600;
    margin: 4px; letter-spacing: 0.5px;
}

.orb {
    position: absolute; border-radius: 50%;
    filter: blur(60px); opacity: 0.15; animation: orbFloat 8s ease-in-out infinite;
}
.orb-1 { width: 200px; height: 200px; background: #00d4ff; top: -50px; left: -50px; animation-delay: 0s; }
.orb-2 { width: 150px; height: 150px; background: #7b2fff; bottom: -30px; right: -30px; animation-delay: -4s; }
.orb-3 { width: 100px; height: 100px; background: #0055aa; top: 50%; left: 50%; animation-delay: -2s; }

@keyframes shimmer {
    0%   { background-position: 0% center; }
    100% { background-position: 200% center; }
}
@keyframes fadeIn {
    from { opacity: 0; } to { opacity: 1; }
}
@keyframes popIn {
    from { opacity: 0; transform: scale(0.8); }
    to   { opacity: 1; transform: scale(1); }
}
@keyframes slideInLeft {
    from { opacity: 0; transform: translateX(-20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes gradientShift {
    0%   { background-position: 0% center; }
    50%  { background-position: 100% center; }
    100% { background-position: 0% center; }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 10px rgba(0,212,255,0.3); }
    50%       { box-shadow: 0 0 25px rgba(0,212,255,0.6), 0 0 50px rgba(123,47,255,0.2); }
}
@keyframes rotateBg {
    from { transform: rotate(0deg); }
    to   { transform: rotate(360deg); }
}
@keyframes orbFloat {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-20px); }
}
@keyframes shineSwipe {
    0%   { transform: translateX(-100%) rotate(45deg); }
    100% { transform: translateX(100%) rotate(45deg); }
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

# ── Secrets ───────────────────────────────────────────────────────────────────
def get_secret(key, default=""):
    try:
        return st.secrets.get(key, default)
    except:
        return default

DEFAULT_GROQ_KEY     = get_secret("GROQ_API_KEY")
DEFAULT_RAPIDAPI_KEY = get_secret("RAPIDAPI_KEY")

# ── Helpers ───────────────────────────────────────────────────────────────────
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
                'eg','etc','skills','and','or','a','the','of','in','for','with','on','to','from'
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
    end   = text.rfind(']')
    if start != -1 and end != -1:
        try:
            return json.loads(text[start:end+1])
        except:
            pass
    return []

def get_live_jobs(job_title, location, api_key):
    url = "https://jsearch.p.rapidapi.com/search"
    headers = {"X-RapidAPI-Key": api_key, "X-RapidAPI-Host": "jsearch.p.rapidapi.com"}
    params  = {"query": f"{job_title} in {location}"}
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
        with open(os.path.join(kaggle_dir, 'kaggle.json'), 'w') as f:
            json.dump({"username": st.secrets["kaggle"]["username"],
                       "key": st.secrets["kaggle"]["key"]}, f)
        os.chmod(os.path.join(kaggle_dir, 'kaggle.json'), 0o600)
        import kaggle
        kaggle.api.authenticate()
        with st.spinner("📥 Downloading dataset from Kaggle..."):
            kaggle.api.dataset_download_files(
                'bcsf24m006asadullah/job-desciptions', path='.', unzip=True)
        return os.path.exists('job_descriptions.csv')
    except Exception as e:
        st.warning(f"⚠️ Kaggle download failed: {e}. Using job_dataset.csv only.")
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
    except:
        job_skills_df = pd.DataFrame(columns=['Job Title', 'skills'])
    try:
        df2 = pd.read_csv('job_dataset.csv', engine='python', on_bad_lines='skip')
        df2_s = df2[['Title', 'Keywords']].copy()
        df2_s.rename(columns={'Title': 'Job Title', 'Keywords': 'skills'}, inplace=True)
    except:
        df2_s = pd.DataFrame(columns=['Job Title', 'skills'])

    combined = pd.concat([job_skills_df, df2_s], ignore_index=True).dropna()
    for col in ['Job Title', 'skills']:
        combined[col] = combined[col].apply(to_lowercase)
        combined[col] = combined[col].apply(remove_punctuation)
        combined[col] = combined[col].apply(remove_brackets)

    job_title_to_skills = (
        combined.groupby('Job Title')['skills']
        .apply(lambda x: ' '.join(x.dropna().unique()))
        .reset_index()
    )
    unique_titles    = job_title_to_skills['Job Title'].tolist()
    combined_skills  = job_title_to_skills['skills'].tolist()
    all_skills       = [s for sub in combined['skills'].apply(advanced_tokenize_skills) for s in sub]
    job_skills_set   = set(all_skills)

    embedder   = load_embedder()
    embeddings = embedder.encode(combined_skills, convert_to_tensor=False).astype('float32')
    index      = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return combined, job_title_to_skills, unique_titles, job_skills_set, index

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="orb orb-1"></div>
    <div class="orb orb-2"></div>
    <div class="orb orb-3"></div>
    <span class="glow-title">🤖 AI Career Mentor</span>
    <p class="subtitle-text">
        Upload your resume · Discover your ideal career path<br>
        Get a personalized roadmap · Find live job opportunities
    </p>
    <br>
    <span class="badge">⚡ Groq LLM</span>
    <span class="badge">🔍 FAISS Semantic Search</span>
    <span class="badge">📊 1.6M+ Job Descriptions</span>
    <span class="badge">🌐 Live Job Search</span>
</div>
""", unsafe_allow_html=True)

# ── How it works ──────────────────────────────────────────────────────────────
with st.expander("✨ How does it work?"):
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("""
        <div class="step-card">
            <div class="step-icon">📄</div>
            <div class="step-title">Upload Resume</div>
            <div class="step-desc">Drop your PDF resume and let AI do the rest</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="step-card">
            <div class="step-icon">🧠</div>
            <div class="step-title">Skill Extraction</div>
            <div class="step-desc">Groq LLM extracts all your technical & soft skills</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="step-card">
            <div class="step-icon">⚡</div>
            <div class="step-title">Career Matching</div>
            <div class="step-desc">FAISS matches you to the best career paths</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown("""
        <div class="step-card">
            <div class="step-icon">🚀</div>
            <div class="step-title">Your Roadmap</div>
            <div class="step-desc">Get a phase-by-phase plan + live job listings</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    location = st.text_input("📍 Job Search Location", value="Lahore, Pakistan")
    top_k    = st.slider("🎯 Career Recommendations", 3, 10, 5)
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; padding: 1rem;
         background: linear-gradient(135deg, #0d1117, #111827);
         border: 1px solid #1e2a3a; border-radius: 12px;'>
        <div style='font-size:1.5rem; margin-bottom:0.5rem;'>🔒</div>
        <div style='color:#00d4ff; font-weight:600; font-size:0.9rem;'>Secured Platform</div>
        <div style='color:#6b7280; font-size:0.78rem; margin-top:0.4rem; line-height:1.5;'>
            API keys are managed securely.<br>No setup required.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#6b7280; font-size:0.82rem;'>
        Built with ❤️ by<br>
        <a href='https://asadullahjvd.github.io' style='color:#00d4ff; font-weight:600;'>
        Asadullah Javed</a>
    </div>
    """, unsafe_allow_html=True)

# ── Keys from secrets ─────────────────────────────────────────────────────────
groq_api_key = DEFAULT_GROQ_KEY
rapidapi_key = DEFAULT_RAPIDAPI_KEY

# ── Session state ─────────────────────────────────────────────────────────────
for key, val in {
    'clicked_recommend': False,
    'extracted_skills': [],
    'extracted_skills_lower': [],
    'recommended_careers': []
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Input ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("📄 Upload Your Resume")
    uploaded_file = st.file_uploader("Upload PDF resume", type="pdf")
with col2:
    st.subheader("💡 Your Interests")
    user_interests = st.text_area(
        "What excites you?",
        placeholder="e.g. AI, Machine Learning, Web Development...",
        height=133
    )

st.markdown("---")

# ── Button ────────────────────────────────────────────────────────────────────
if st.button("🚀 Get Career Recommendations", use_container_width=True):
    if not uploaded_file:
        st.error("⚠️ Please upload your resume PDF.")
        st.stop()
    if not groq_api_key:
        st.error("⚠️ API key not configured. Please contact the administrator.")
        st.stop()

    try:
        with st.spinner("🔍 Loading career database..."):
            combined, job_title_to_skills, unique_titles, job_skills_set, faiss_index = load_data_and_index()
            embedder = load_embedder()

        if len(unique_titles) == 0:
            st.error("Career dataset could not be compiled.")
            st.stop()

        with st.spinner("📖 Reading your resume..."):
            resume_text = extract_text_from_pdf(io.BytesIO(uploaded_file.read()))
            if not resume_text or len(resume_text.strip()) == 0:
                st.error("Could not extract text. Is your PDF a scanned image?")
                st.stop()

        with st.spinner("🧠 Extracting skills with Groq LLM..."):
            llm    = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")
            prompt = f"""
You are a skill extraction system. Extract ALL technical and soft skills from the resume below.
Return ONLY a valid JSON array of skill names. No markdown, no explanation.
Example: ["Python", "Machine Learning", "SQL", "Communication"]

Resume:
{resume_text}
"""
            response    = llm.invoke(prompt)
            raw_content = response.content.strip()
            extracted   = extract_json_skills(raw_content)

            if not extracted and '[' in raw_content and ']' in raw_content:
                try:
                    extracted = json.loads(raw_content[raw_content.find('['):raw_content.find(']')+1])
                except:
                    pass

            if not extracted:
                extracted = [s.strip() for s in re.sub(r'[\[\]\"\']', '', raw_content).split(',') if s.strip()]

            if not extracted:
                st.error("Skill extraction failed. Please try again.")
                st.stop()

            st.session_state.extracted_skills       = extracted
            st.session_state.extracted_skills_lower = [s.lower() for s in extracted]

        with st.spinner("⚡ Matching careers with FAISS..."):
            cv_embedding = embedder.encode(
                ", ".join(st.session_state.extracted_skills_lower),
                convert_to_tensor=False
            ).astype('float32')
            distances, indices = faiss_index.search(np.array([cv_embedding]), top_k)

        recommended_careers = []
        user_skills_set     = set(st.session_state.extracted_skills_lower)

        for i in range(top_k):
            if i >= len(indices[0]):
                break
            idx = indices[0][i]
            if idx >= len(unique_titles):
                continue
            career_title = unique_titles[idx]
            similarity   = round(1 - (distances[0][i] / 2), 4)
            skills_match = job_title_to_skills[job_title_to_skills['Job Title'] == career_title]
            missing_skills = sorted(
                set(advanced_tokenize_skills(skills_match['skills'].iloc[0])) - user_skills_set
            ) if not skills_match.empty else []

            recommended_careers.append({
                "title": career_title.title(),
                "similarity": similarity,
                "missing_skills": missing_skills
            })

        st.session_state.recommended_careers = recommended_careers
        st.session_state.clicked_recommend   = True

    except Exception as e:
        st.error(f"Error: {e}")

# ── Results ───────────────────────────────────────────────────────────────────
if st.session_state.clicked_recommend and st.session_state.extracted_skills:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

    st.markdown("<div class='section-header'><h3>🛠️ Skills Extracted from Your Resume</h3></div>", unsafe_allow_html=True)
    st.markdown(
        "".join([f"<span class='skill-tag'>{s}</span>" for s in st.session_state.extracted_skills]),
        unsafe_allow_html=True
    )

    st.markdown("<div class='section-header'><h3>🎯 Recommended Career Paths</h3></div>", unsafe_allow_html=True)
    for i, career in enumerate(st.session_state.recommended_careers):
        with st.expander(f"{i+1}. {career['title']} — Match: {career['similarity']:.2%}"):
            if career['missing_skills']:
                st.markdown("**❌ Skills to develop:**")
                st.markdown(
                    "".join([f"<span class='missing-tag'>{s}</span>" for s in career['missing_skills'][:15]]),
                    unsafe_allow_html=True
                )
            else:
                st.success("✅ You have all major skills for this role!")

    st.markdown("<div class='section-header'><h3>🗺️ Learning Roadmap</h3></div>", unsafe_allow_html=True)
    career_options = [c['title'] for c in st.session_state.recommended_careers]

    if career_options:
        selected_career = st.selectbox("Select a career path:", career_options)

        if selected_career:
            selected_info = next((c for c in st.session_state.recommended_careers if c['title'] == selected_career), None)
            if selected_info:
                missing = selected_info['missing_skills']

                with st.spinner("✍️ Generating your personalized roadmap..."):
                    try:
                        roadmap_response = llm.invoke(f"""
You are an expert career advisor. The user wants to become a {selected_career}.
Current skills: {', '.join(st.session_state.extracted_skills_lower[:20])}
Interests: {user_interests}
Missing skills: {', '.join(missing[:15]) if missing else 'None — well prepared!'}

Generate a clear, structured learning roadmap with:
1. Phase-by-phase plan (Phase 1, 2, 3)
2. Specific resources/courses per missing skill
3. Estimated timeline per phase
4. Final tip for landing first job/internship
Format as clean markdown.
""")
                        st.markdown(roadmap_response.content)
                    except Exception as e:
                        st.error(f"Roadmap generation failed: {e}")

                st.markdown("<div class='section-header'><h3>💼 Live Job Opportunities</h3></div>", unsafe_allow_html=True)
                if rapidapi_key:
                    with st.spinner(f"🔎 Searching live jobs for {selected_career}..."):
                        jobs = get_live_jobs(selected_career, location, rapidapi_key)
                    if jobs:
                        for job in jobs[:8]:
                            st.markdown(f"""
<div class='job-card'>
    <strong style='color:#e0e0e0; font-size:1rem;'>{job.get('job_title','N/A')}</strong><br>
    🏢 <span style='color:#a0aec0;'>{job.get('employer_name','N/A')}</span> &nbsp;|&nbsp;
    📍 <span style='color:#a0aec0;'>{job.get('job_city', location)}</span> &nbsp;|&nbsp;
    📌 <span style='color:#a0aec0;'>{job.get('job_employment_type','')}</span><br><br>
    <a href='{job.get('job_apply_link','#')}' target='_blank'>👉 Apply Now</a>
</div>""", unsafe_allow_html=True)
                    else:
                        st.info("No live jobs found. Try a different location.")
                else:
                    st.info("Live job search not configured.")

                with st.spinner("📊 Generating job search guidance..."):
                    try:
                        guidance_response = llm.invoke(f"""
You are a job search advisor. Fresh CS graduate targeting: {selected_career} in {location}.
Provide actionable guidance in clean markdown:
1. **Top Search Queries** — 3-5 queries for LinkedIn/Indeed
2. **Best Platforms** — where to find jobs in Pakistan/remotely
3. **Key Companies** — who hires for this role in Pakistan
4. **Quick Tips** — 3 tips to stand out as a fresher
Keep it concise and practical.
""")
                        st.markdown("#### 🔍 Job Search Guidance")
                        st.markdown(guidance_response.content)
                    except Exception as e:
                        st.error(f"Guidance generation failed: {e}")
