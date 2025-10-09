import re
from io import BytesIO
import logging
import spacy
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import PyPDF2 
import docx 
import os 

# --- Configuration & Setup (Unchanged) ---
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
SEMANTIC_MODEL_NAME = "all-mpnet-base-v2" 
SPACY_MODEL_NAME = "en_core_web_sm" 

nlp = None
embedding_model = None

try:
    nlp = spacy.load(SPACY_MODEL_NAME) 
    embedding_model = SentenceTransformer(SEMANTIC_MODEL_NAME)
    logging.info(f"Loaded Spacy model: {SPACY_MODEL_NAME} and SentenceTransformer model: {SEMANTIC_MODEL_NAME}")
except OSError as e:
    logging.error(f"Model loading failed. Ensure '{SPACY_MODEL_NAME}' is downloaded and files are accessible. Error: {e}")
    nlp = None
    embedding_model = None


# --- Constants (Unchanged) ---
SEMANTIC_WEIGHT = 0.1 
KEYWORD_WEIGHT = 0.9 
SCORE_SCALING_FACTOR = 3.55 
MATCH_THRESHOLD = 65 
WORK_EXPERIENCE_HEADINGS = [
    "work experience", "professional experience", "employment history", 
    "relevant experience", "experience", "job history"
]
EDUCATION_HEADINGS = [
    "education", "academic history", "educational qualifications", "qualifications"
]
# ------------------------------------------

# --- Existing Functions (Text Extraction, Candidate Info, Experience, Scoring) ---

def extract_text(file_obj):
    # ... (function body remains the same)
    text = ""
    fname = os.path.basename(getattr(file_obj, 'name', 'unknown_file')).lower()
    
    try:
        file_obj.seek(0)
        
        if fname.endswith(".pdf"):
            reader = PyPDF2.PdfReader(file_obj)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text: text += page_text + "\n"
        elif fname.endswith(".docx"):
            doc = docx.Document(file_obj)
            for para in doc.paragraphs:
                text += para.text + "\n"
        else:
            text = file_obj.read().decode('utf-8', errors='ignore')
            
    except Exception as e:
        logging.error(f"Text extraction failed for {fname}: {e}")
        return ""
        
    return re.sub(r'[^\S\r\n]+', ' ', text).strip()

def is_header_or_title(text, job_role=""):
    # ... (function body remains the same)
    clean_text = text.strip().lower()
    
    common_headers_and_titles = {
        "resume", "curriculum vitae", "contact", "experience", "education",
        "profile", "summary", "objective", "skills", "declaration", "project",
        "career objective", "career summary", "references", "developer", 
        "engineer", "manager", "architect", "analyst", "specialist", "portfolio"
    }
    
    if job_role:
        common_headers_and_titles.add(job_role.lower())
        
    if len(clean_text.split()) > 5 or any(c.isdigit() or c in '()[]{}|@' for c in clean_text): 
        return True 
    
    if any(header in clean_text for header in common_headers_and_titles):
        if clean_text in common_headers_and_titles or job_role.lower() in clean_text:
             return True
            
    return False

import os, re

def extract_candidate_info(text, job_role="", resume_filename=""):
    """
    Extract full name, email, and phone number from resume text.
    Strong preference for real human-looking names.
    Fallback: derive clean name from email username.
    """

    # ------------------ EMAIL ------------------
    email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
    email = email_match.group(0).strip() if email_match else "N/A"

    # ------------------ PHONE ------------------
    phone_match = re.search(
        r'(\+?\d{1,3}[\s\-\.]?)?\(?\d{2,4}\)?[\s\-\.]?\d{3,4}[\s\-\.]?\d{3,4}', text
    )
    phone = phone_match.group(0).strip() if phone_match else "N/A"

    # ------------------ TEXT CLEANUP ------------------
    lines = [re.sub(r'[^A-Za-z\s]', ' ', line).strip() for line in text.split("\n")]
    lines = [line for line in lines if line]

    # Words that *cannot* be a name
    banned_words = {
        "python", "django", "flask", "developer", "engineer", "java", "sql", "html",
        "css", "javascript", "react", "node", "git", "github", "aws", "docker",
        "api", "resume", "curriculum", "vitae", "profile", "objective", "skills",
        "education", "experience", "project"
    }

    probable_name = "N/A"

    # ------------------ STEP 1: spaCy PERSON ------------------
    if nlp is not None:
        doc = nlp(" ".join(lines[:50]))
        persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON"]
        if persons:
            probable_name = persons[0]

    # ------------------ STEP 2: Manual header scan ------------------
    if probable_name == "N/A":
        for line in lines[:15]:
            if any(w.lower() in banned_words for w in line.lower().split()):
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w.isalpha()):
                probable_name = " ".join(words)
                break

    # ------------------ STEP 3: Validate name ------------------
    if (
        probable_name == "N/A"
        or len(probable_name) < 3
        or any(w.lower() in banned_words for w in probable_name.lower().split())
    ):
        probable_name = "N/A"

    # ------------------ STEP 4: Email-based fallback ------------------
    if probable_name == "N/A" and email != "N/A":
        username = email.split("@")[0]
        username = re.sub(r'[\d\._\-]+', ' ', username)
        username = re.sub(r'\s+', ' ', username).strip()

        parts = [p.capitalize() for p in username.split() if len(p) > 1]

        # if single chunk like "chiragmodi" → split mid into two
        if len(parts) == 1 and len(parts[0]) > 6:
            halves = re.findall(r'[A-Z]?[a-z]+', parts[0]) or [parts[0][:5], parts[0][5:]]
            probable_name = " ".join(h.capitalize() for h in halves[:2])
        elif len(parts) >= 2:
            probable_name = " ".join(parts[:2])
        elif len(parts) == 1:
            probable_name = parts[0]

    # ------------------ STEP 5: Filename fallback ------------------
    if probable_name == "N/A" and resume_filename:
        base = os.path.basename(resume_filename)
        base = re.sub(r'[_\-\.]', ' ', os.path.splitext(base)[0])
        probable_name = " ".join(w.capitalize() for w in base.split()[:2])

    # ------------------ FINAL CLEANUP ------------------
    probable_name = re.sub(r'\s+', ' ', probable_name).strip()
    if any(w.lower() in banned_words for w in probable_name.lower().split()):
        probable_name = "N/A"

    return probable_name or "N/A", email, phone


def extract_date_ranges(text):
    # ... (function body remains the same)
    date_patterns = [
        r'(\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?\s+\d{4})\s*[\-–]\s*(\b(?:Present|Current|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*[.,]?\s*\d{4}|\bPresent|\bCurrent)',
        r'(\d{4})\s*[\-–]\s*(\d{4}|\bPresent|\bCurrent)',
        r'(\b\d{1,2}/\d{4})\s*[\-–]\s*(\b\d{1,2}/\d{4}|\bPresent|\bCurrent)'
    ]
    
    date_ranges = []
    
    def parse_date(date_string):
        date_string = date_string.replace('Current', '').replace('Present', '').strip()
        if not date_string:
            return datetime.now()
        
        formats = ["%b %Y", "%B %Y", "%m/%Y", "%Y"]
        for fmt in formats:
            try:
                if len(date_string.split()) > 1 and date_string.split()[0].isalpha():
                    return datetime.strptime(date_string.split()[0][:3] + " " + date_string.split()[-1], "%b %Y")
                elif re.match(r'\d{4}', date_string) and len(date_string) == 4:
                    return datetime.strptime(date_string, "%Y")
                else:
                    return datetime.strptime(date_string, fmt)
            except ValueError:
                continue
        return None

    for pattern in date_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for start_str, end_str in matches:
            start_date = parse_date(start_str)
            end_date = parse_date(end_str)
            
            if start_date and end_date and start_date < end_date:
                date_ranges.append((start_date, end_date))
                
    return date_ranges

def get_work_experience_text(resume_text):
    # ... (function body remains the same)
    text_lines = resume_text.split('\n')
    experience_text = []
    in_experience_section = False
    
    header_pattern = re.compile(r'^\s*([A-Z\s]{4,}|[A-Z][a-z]+ [A-Z][a-z]+)\s*$')

    for line in text_lines:
        line_lower = line.strip().lower()
        is_header = header_pattern.match(line)
        
        if not in_experience_section:
            if any(h in line_lower for h in WORK_EXPERIENCE_HEADINGS) and len(line_lower.split()) <= 4:
                in_experience_section = True
                continue 

        if in_experience_section:
            if is_header and any(h in line_lower for h in EDUCATION_HEADINGS):
                break
            
            experience_text.append(line)

    return "\n".join(experience_text)

def calculate_duration_in_years_and_months(start_date, end_date):
    # ... (function body remains the same)
    diff = end_date - start_date
    total_months = diff.days // 30 
    years = total_months // 12
    months = total_months % 12
    return years, months

import re
from datetime import datetime

# --------------------- Helper functions ---------------------

def parse_date_str(date_str):
    """Convert fuzzy month-year strings into datetime objects."""
    date_str = date_str.strip().replace(".", "")
    today = datetime.today()

    # If mentions "present" or "current", return today's date
    if re.search(r'present|current|till date|now', date_str, re.IGNORECASE):
        return today

    patterns = [
        ("%b %Y", r"^[A-Za-z]{3,}\s+\d{4}$"),  # Jan 2020
        ("%B %Y", r"^[A-Za-z]{3,}\s+\d{4}$"),  # January 2020
        ("%m/%Y", r"^\d{1,2}/\d{4}$"),         # 05/2020
        ("%Y", r"^\d{4}$"),                    # 2020
    ]

    for fmt, regex in patterns:
        try:
            return datetime.strptime(date_str, fmt)
        except Exception:
            continue
    return None


def extract_date_ranges(text):
    """Extract all date ranges from the resume text."""
    # Common range patterns
    date_pattern = (
        r"(?P<start>(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2}[\/\-]\d{4}|\d{4}))"
        r"\s*(?:to|\-|–|—|until|till|upto)\s*"
        r"(?P<end>(?:Present|Current|Now|Till Date|"
        r"Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
        r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
        r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|\d{1,2}[\/\-]\d{4}|\d{4}))"
    )

    matches = re.findall(date_pattern, text, re.IGNORECASE)
    ranges = []

    for start_str, end_str in matches:
        start_date = parse_date_str(start_str)
        end_date = parse_date_str(end_str)
        if start_date and end_date and start_date <= end_date:
            ranges.append((start_date, end_date))
    return ranges


def calculate_duration_in_years_and_months(start, end):
    """Return duration between two datetimes as (years, months)."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    years = months // 12
    rem_months = months % 12
    return years, rem_months

# --------------------- Main Function ---------------------

def extract_total_experience(resume_text):
    work_exp_text = resume_text  # you can call get_work_experience_text(resume_text) if needed
    date_ranges = extract_date_ranges(work_exp_text)

    if not date_ranges:
        return 0.0, "N/A"

    # Sort & merge overlapping ranges
    date_ranges.sort(key=lambda x: x[0])
    merged = []
    cur_start, cur_end = date_ranges[0]

    for next_start, next_end in date_ranges[1:]:
        if next_start <= cur_end:
            cur_end = max(cur_end, next_end)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = next_start, next_end
    merged.append((cur_start, cur_end))

    # Total duration
    total_months = 0
    for s, e in merged:
        y, m = calculate_duration_in_years_and_months(s, e)
        total_months += y * 12 + m

    total_years = total_months / 12
    y_int = int(total_years)
    m_int = int(round((total_years - y_int) * 12))
    experience_str = f"{y_int} Years, {m_int} Months"

    return round(total_years, 1), experience_str

def extract_required_experience(jd_text):
    # ... (function body remains the same)
    matches = re.findall(r'(\d+)(?:\s*[\+\-]\s*\d+)?\s*(?:to)?\s*years?', jd_text, re.IGNORECASE)
    
    if matches:
        min_exp = min(int(match[0]) for match in matches)
        return float(min_exp)
    
    min_match = re.search(r'(?:minimum|min|at least)\s*(\d+)\s*years?', jd_text, re.IGNORECASE)
    if min_match:
        return float(min_match.group(1))

    return 0.0 

def experience_match_score(candidate_years, required_years):
    # ... (function body remains the same)
    if required_years <= 0:
        return 100.0, "No Requirement Found"
        
    if candidate_years >= required_years:
        return 100.0, "Requirement Met or Exceeded"
    else:
        score = (candidate_years / required_years) * 100
        return round(min(score, 99.0), 1), f"Short by {round(required_years - candidate_years, 1)} years"

def semantic_similarity(resume_text, jd_text):
    # ... (function body remains the same)
    if nlp is None or embedding_model is None or not jd_text or not resume_text:
        return 0.0
    
    try:
        def filter_sentences(text):
            sents = [sent.text.lower() for sent in nlp(text).sents 
                     if len(sent.text.strip()) > 10 and any(c.isalpha() for c in sent.text)]
            return sents

        resume_sents = filter_sentences(resume_text)
        jd_sents = filter_sentences(jd_text)
        
        if not resume_sents or not jd_sents:
            logging.warning("Insufficient sentences for semantic analysis after filtering.")
            return 0.0

        emb_resume = embedding_model.encode(resume_sents, convert_to_tensor=True)
        emb_jd = embedding_model.encode(jd_sents, convert_to_tensor=True)
        
        cos_scores = util.cos_sim(emb_resume, emb_jd)
        
        max_scores_per_jd_sent = cos_scores.max(dim=0).values
        
        semantic_score = float(max_scores_per_jd_sent.mean() * 100)
        
        return round(semantic_score, 2)
        
    except Exception as e:
        logging.warning(f"Semantic similarity failed: {e}")
        return 0.0

def get_keywords(text):
    # ... (function body remains the same)
    if nlp is None: return set()

    doc = nlp(text.lower())
    keywords = set()
    
    irrelevant = {"etc", "description", "cv", "ltd", "solution", "university", "ability", "requirement", "pvt"}
    
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if 2 <= len(phrase.split()) <= 5 and len(phrase) > 3 and phrase not in irrelevant:
            keywords.add(phrase)
            
    for token in doc:
        if token.pos_ in ('NOUN', 'PROPN') and token.is_alpha and not token.is_stop and len(token.text) > 2:
            if token.text not in irrelevant:
                keywords.add(token.text)
            
    return keywords

def keyword_match_score(resume_text, jd_text):
    # ... (function body remains the same)
    jd_keywords = get_keywords(jd_text)
    resume_keywords = get_keywords(resume_text)
    
    if not jd_keywords:
        return 0.0

    matched_keywords = jd_keywords.intersection(resume_keywords)
    score = (len(matched_keywords) / len(jd_keywords)) * 100
    
    logging.info(f"JD Keywords extracted ({len(jd_keywords)}): {list(jd_keywords)[:5]}...")
    logging.info(f"Matched Keywords ({len(matched_keywords)}): {list(matched_keywords)[:5]}...")

    return round(score, 2)
    
# ----------------- NEW ALIGNMENT FUNCTION -----------------

def generate_strategic_alignment(aggregate_score, candidate_years, required_years, sem_score, keyword_score):
    """
    Creates the strategic, actionable summary based on all calculated metrics.
    """
    alignment = []

    # 1. Overall Recommendation
    if aggregate_score >= 85:
        alignment.append("✅ **High Priority Candidate:** The overall fitment score is excellent. Proceed directly to the interview stage.")
    elif aggregate_score >= MATCH_THRESHOLD: # 65
        alignment.append("🔶 **Qualified Candidate:** The score indicates a strong baseline match. Requires a screening interview to validate experience depth.")
    else:
        alignment.append("🛑 **Low Priority Candidate:** The score is below the threshold. Reroute only if the talent pipeline is scarce.")

    # 2. Experience Risk
    if required_years > 0 and candidate_years < required_years * 0.8:
        gap = round(required_years - candidate_years, 1)
        alignment.append(f"⚠️ **Experience Gap:** Candidate is short by {gap} years. Focus interview questions on transferable skills and project ownership to compensate.")
    elif candidate_years >= required_years:
        alignment.append("🟢 **Experience Met:** Candidate meets or exceeds the required professional experience.")
    else:
         alignment.append("🔵 **Experience Context:** No explicit experience requirement found in JD, or candidate is close to target.")
    
    # 3. Score Breakdown Insight
    if sem_score > keyword_score * 2:
        alignment.append("💡 **Semantic Strength:** The candidate's *concept* of the role aligns well (high semantic score). They may use different terminology than the JD. Review project descriptions closely.")
    elif keyword_score > sem_score * 1.5:
        alignment.append("💡 **Keyword Focus:** The resume is highly optimized for keywords. Ensure the candidate can articulate their experience (low semantic score suggests a potential knowledge gap beneath the surface.)")
    else:
        alignment.append("✅ **Balanced Profile:** Strong, consistent alignment in both skills (keywords) and conceptual understanding (semantic).")
        
    return "\n- " + "\n- ".join(alignment)


# ----------------- Overview Generation (Unchanged) -----------------
def generate_candidate_overview(name, job_role, aggregate_score, sem_score, keyword_score, recommendation, exp_match_score, req_exp):
    """Generates a descriptive, data-driven summary of the candidate's fit."""
    
    fit_level = ""
    if aggregate_score >= 90:
        fit_level = "an Exceptional Match"
        strength = "The alignment is near-perfect, indicating a strong cultural and technical fit."
    elif aggregate_score >= 80:
        fit_level = "a Strong Match"
        strength = "The candidate possesses a high degree of core competency and is immediately hirable."
    elif aggregate_score >= 65:
        fit_level = "a Moderate to Strong Match"
        strength = "The core skills are present, but some key areas may require further evaluation."
    else:
        fit_level = "a Borderline Match"
        strength = "The profile meets minimum conceptual requirements but lacks significant keyword depth."

    if sem_score > keyword_score * 2:
        match_type = "The high Semantic Score suggests strong conceptual understanding and relevant experience, even if specific keywords were not explicitly matched."
    elif keyword_score > sem_score * 1.5:
        match_type = "The profile is heavily keyword-optimized, but conceptual alignment is a concern. A manual review is essential to verify experience depth."
    else:
        match_type = "The conceptual and keyword matches are balanced, providing a consistent view of the candidate's profile."
        
    candidate_identifier = f"Mr./Ms. {name}" if name != "N/A" else "The Candidate"

    exp_context = ""
    if req_exp > 0 and exp_match_score < 100.0:
        exp_context = f"**Experience Alert:** Candidate only meets {exp_match_score:.1f}% of the required {req_exp} years of experience. "
    elif req_exp > 0:
        exp_context = f"Experience requirement of {req_exp} years is met. "


    overview = (
        f"{exp_context}"
        f"{candidate_identifier} is currently rated as {fit_level} for the {job_role} position "
        f"with an Aggregate Score of {aggregate_score:.2f}%. {strength} "
        f"The system recommends: {recommendation}. {match_type}"
    )
    return overview


# ----------------- ATS Analyzer (Main Function) -----------------
def analyze_resume(resume_file_obj, job_description_file_obj, job_role="general role"):
    # ... (rest of the function remains the same, but includes the new field)
    if nlp is None or embedding_model is None:
        return {"full_name":"N/A","contact_number":"N/A","email":"N/A",
                "job_role": job_role, "aggregate_score":0.0,"fitment_verdict":"FATAL ERROR",
                "hiring_recommendation":"Not Applicable", "overall_experience": "N/A", 
                "experience_match": "N/A", "strategic_alignment": "FATAL ERROR: Models failed to load.",
                "analysis_summary":"NLP/Embedding models failed to load. Cannot proceed.",
                "candidate_overview": "Model loading failed."}


    logging.info("Starting text extraction...")
    resume_text = extract_text(resume_file_obj)
    jd_text = extract_text(job_description_file_obj)
    
    if not resume_text or not jd_text: 
        return {"full_name":"N/A","contact_number":"N/A","email":"N/A",
                "job_role": job_role, "aggregate_score":0.0,"fitment_verdict":"ERROR",
                "hiring_recommendation":"Not Applicable", "overall_experience": "N/A", 
                "experience_match": "N/A", "strategic_alignment": "ERROR: Text extraction failed.",
                "analysis_summary":"Failed to extract text from one or both files.",
                "candidate_overview": "Failed to extract necessary text for analysis."}

    # 1. Extract Candidate Info
    name, email, phone = extract_candidate_info(resume_text, job_role) 
    
    # 2. Extract and Score Experience
    candidate_years, candidate_exp_str = extract_total_experience(resume_text)
    required_years = extract_required_experience(jd_text)
    
    exp_match_score_val, exp_match_verdict = experience_match_score(candidate_years, required_years)
    
    overall_experience_display = candidate_exp_str
    experience_match_display = f"{exp_match_score_val}% ({exp_match_verdict}) [Req: {required_years} Yrs]"
    
    # 3. Calculate Scores (Semantic and Keyword)
    sem_score = semantic_similarity(resume_text, jd_text)
    keyword_score = keyword_match_score(resume_text, jd_text)


    # 4. Aggregate Score
    weighted_score = (sem_score * SEMANTIC_WEIGHT) + (keyword_score * KEYWORD_WEIGHT)
    aggregate_score = round(min(weighted_score * SCORE_SCALING_FACTOR, 100.0), 2)
    
    logging.info(f"Final Scaled Score: {aggregate_score}%")

    # 5. Determine Verdict and Recommendation
    if aggregate_score >= 85: 
        verdict = "Selected / High Match"
        recommendation = "Hire Recommended"
    elif aggregate_score >= MATCH_THRESHOLD: # 65
        verdict = "Borderline / Review"
        recommendation = "Manual Review Recommended"
    else:
        verdict = "Fail / Not Selected"
        recommendation = "Not Recommended"

    # 6. Generate New Strategic Alignment
    strategic_alignment = generate_strategic_alignment(
        aggregate_score, candidate_years, required_years, sem_score, keyword_score
    )

    # 7. Generate Descriptive Overview
    overview = generate_candidate_overview(
        name, job_role, aggregate_score, sem_score, keyword_score, 
        recommendation, exp_match_score_val, required_years
    )

    return {
        "full_name": name,
        "contact_number": phone,
        "email": email,
        "job_role": job_role,
        "overall_experience": overall_experience_display, 
        "experience_match": experience_match_display,     
        "aggregate_score": aggregate_score,
        "fitment_verdict": verdict,
        "hiring_recommendation": recommendation,
        "analysis_summary": (
            f"Semantic Score: {sem_score}%; Keyword Score: {keyword_score}%; "
            f"Final Scaled Score: {aggregate_score}%"
        ),
        "strategic_alignment": strategic_alignment, # NEW FIELD
        "candidate_overview": overview, 
    }

# ----------------- Example Usage (Updated Print Logic) -----------------
if __name__ == '__main__':
    # !!! --- IMPORTANT: UPDATE THESE FILE PATHS TO MATCH YOUR LOCAL FILES --- !!!
    JD_FILE_PATH = r"C:\Users\rahul.suthar\Downloads\JD-202402-Python-Developer.pdf"
    RESUME_FILE_PATH = r"C:\Users\rahul.suthar\Downloads\Chirag Modi (1) (1).pdf"
    JOB_TITLE = "Python Developer"
    
    print("--- ATS Analysis Tool Initializing ---")
    
    if nlp is None or embedding_model is None:
        print("\nFATAL ERROR: Model dependency failed to load. Please fix the OSError before running.")
    else:
        try:
            # Open files in binary mode
            with open(RESUME_FILE_PATH, 'rb') as resume_f, \
                 open(JD_FILE_PATH, 'rb') as jd_f:
                
                results = analyze_resume(resume_f, jd_f, job_role=JOB_TITLE)
                
                print("\n" + "="*50)
                print(" " * 15 + "ATS ANALYSIS REPORT")
                print("="*50)

                # Define the order and format for printing
                print_order = [
                    ("Full Name", "full_name"),
                    ("Email", "email"),
                    ("Contact Number", "contact_number"),
                    ("Job Role", "job_role"),
                    ("--------------------------", ""), # Separator
                    ("Overall Experience", "overall_experience"),
                    ("Experience Match", "experience_match"),
                    ("Aggregate Score", "aggregate_score"),
                    ("Fitment Verdict", "fitment_verdict"),
                    ("Hiring Recommendation", "hiring_recommendation"),
                    ("--------------------------", ""), # Separator
                    ("Analysis Summary", "analysis_summary"),
                    ("Candidate Overview", "candidate_overview"),
                    ("Strategic Alignment", "strategic_alignment"),
                ]

                for label, key in print_order:
                    if key == "":
                        print(label)
                    else:
                        value = results.get(key)
                        if key in ["candidate_overview", "strategic_alignment", "analysis_summary"]:
                            # Print multiline/long fields cleanly
                            print(f"\n{label.upper()}:")
                            if key == "analysis_summary":
                                # Format the Analysis Summary as requested
                                # e.g., Semantic Score: 46.63% | Keyword Score: 20.16% | Final Scaled Score: 80.96%
                                parts = value.split(';')
                                formatted_summary = ' | '.join([p.strip() for p in parts])
                                print(f"  {formatted_summary}")
                            else:
                                print(f"{value}")
                        elif isinstance(value, float):
                            print(f"{label:<22}: {value:.2f}%")
                        else:
                            print(f"{label:<22}: {value}")

        except FileNotFoundError:
            print(f"\nERROR: One or both files not found.")
            print(f"Please ensure '{JD_FILE_PATH}' and '{RESUME_FILE_PATH}' exist and paths are correct.")
        except Exception as e:
            print(f"\nAn unexpected error occurred during analysis: {e}")