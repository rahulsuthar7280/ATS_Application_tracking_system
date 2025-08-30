import os
import io
import json
import re
import requests
import pandas as pd
from datetime import datetime
from pypdf import PdfReader, errors as pypdf_errors
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import time
import logging

# Configure logging for better visibility
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Import docx for .docx file handling
try:
    from docx import Document
except ImportError:
    logging.warning("python-docx not installed. DOCX file processing will not be available.")
    Document = None

# OCR Imports
try:
    from PIL import Image
    from pdf2image import convert_from_bytes, PdfPageCountError, PdfSyntaxError
    import pytesseract
    # --- OCR Configuration (IMPORTANT: Adjust this if Tesseract is not in your PATH) ---
    # For Windows, you might need to specify the full path to tesseract.exe, e.g.:
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    # For Linux/macOS, if tesseract is in your PATH, this line might not be strictly necessary.
    # pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract' # Example for macOS Homebrew
    # ---------------------------------------------------------------------------------
except ImportError:
    logging.warning("OCR libraries (Pillow, pdf2image, pytesseract) not installed. Image-based PDF processing will not be available.")
    Image = None
    convert_from_bytes = None
    pytesseract = None

# Load API keys from Django settings
from django.conf import settings

BLANDAI_API_KEY = getattr(settings, 'BLANDAI_API_KEY', None)
GOOGLE_API_KEY = getattr(settings, 'GOOGLE_API_KEY', None)

# Initialize LLM model globally. It will be re-initialized in llm_call if needed.
llm = None
if GOOGLE_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
        logging.info("Global LLM instance initialized successfully.")
    except Exception as e:
        logging.error(f"Failed to initialize global LLM instance: {e}")
        llm = None
else:
    logging.warning("WARNING: GOOGLE_API_KEY not set in Django settings. LLM functionality will be limited.")

# Base URL for the Gemini API (used for direct requests if not using Langchain's invoke)
GEMINI_API_BASE_URL = "[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/)"
GEMINI_MODEL_NAME = "gemini-2.5-flash-preview-05-20"


def extract_text_from_pdf(pdf_file_obj):
    """
    Extracts text from a PDF file object using pypdf.
    If direct extraction fails, attempts OCR fallback using pdf2image and pytesseract.
    """
    extracted_text = ""
    
    # Try direct text extraction first
    try:
        reader = PdfReader(pdf_file_obj)
        if reader.is_encrypted:
            logging.warning("PDF is encrypted. Direct text extraction might fail.")
            # You might add reader.decrypt('password') here if you have a known password
        
        for page in reader.pages:
            extracted_text += page.extract_text() or ""
        
        if extracted_text.strip():
            logging.info(f"Successfully extracted {len(extracted_text)} characters directly from PDF.")
            return extracted_text.strip()
        else:
            logging.warning("Direct PDF text extraction yielded no text. Attempting OCR fallback.")

    except (pypdf_errors.PdfReadError, pypdf_errors.FileTruncatedError) as pdf_err:
        logging.warning(f"pypdf error during direct read: {pdf_err}. Falling back to OCR.")
    except Exception as e:
        logging.warning(f"Unexpected error during direct PDF extraction: {e}. Falling back to OCR.")

    # Fallback to OCR if direct extraction failed or yielded no text
    if convert_from_bytes and pytesseract and Image:
        try:
            # Reset stream pointer for convert_from_bytes
            pdf_file_obj.seek(0) 
            images = convert_from_bytes(pdf_file_obj.read(), dpi=300) # Higher DPI for better OCR
            
            if not images:
                logging.warning("pdf2image could not convert any pages to images.")
                return ""

            ocr_text_parts = []
            for i, image in enumerate(images):
                logging.debug(f"Performing OCR on page {i+1}...")
                page_ocr_text = pytesseract.image_to_string(image)
                if page_ocr_text:
                    ocr_text_parts.append(page_ocr_text.strip())
            
            extracted_text = "\n".join(ocr_text_parts)
            if extracted_text.strip():
                logging.info(f"Successfully extracted {len(extracted_text)} characters from PDF using OCR.")
                return extracted_text.strip()
            else:
                logging.warning("OCR from PDF yielded no text. It might be entirely blank or unreadable images.")

        except pytesseract.TesseractNotFoundError:
            logging.error("Tesseract OCR engine not found. Please install Tesseract and ensure its path is correctly configured.")
        except (PdfPageCountError, PdfSyntaxError) as pdf2img_err:
            logging.error(f"pdf2image error converting PDF to images: {pdf2img_err}. Check Poppler installation.")
        except Exception as e:
            logging.error(f"Unexpected error during OCR process: {e}")
    else:
        logging.warning("OCR libraries are not available. Cannot perform OCR fallback for PDF.")
        
    return "" # Return empty string if all extraction methods fail

def extract_text_from_docx(docx_file_obj):
    """
    Extracts text from a DOCX file object.
    """
    if Document is None:
        logging.error("python-docx is not installed. Cannot process .docx files.")
        return ""
    try:
        document = Document(docx_file_obj)
        text = []
        for paragraph in document.paragraphs:
            text.append(paragraph.text)
        extracted_text = "\n".join(text).strip()
        logging.info(f"Successfully extracted {len(extracted_text)} characters from DOCX.")
        return extracted_text
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        return ""

def extract_text_from_txt(txt_file_obj):
    """
    Extracts text from a plain text file object.
    """
    try:
        extracted_text = txt_file_obj.read().decode('utf-8').strip()
        logging.info(f"Successfully extracted {len(extracted_text)} characters from TXT.")
        return extracted_text
    except Exception as e:
        logging.error(f"Error extracting text from TXT: {e}")
        return ""

def extract_text_from_document(file_obj, filename):
    """
    Extracts text from various document file types based on extension.
    This function handles the file object directly, including seeking to the beginning.
    """
    if not file_obj:
        logging.warning("extract_text_from_document received no file object.")
        return ""

    ext = os.path.splitext(filename)[1].lower()
    
    # Ensure file pointer is at the beginning for reading
    file_obj.seek(0)
    
    logging.info(f"Attempting to extract text from '{filename}' (Extension: {ext}). File size: {file_obj.size} bytes.")

    try:
        if ext == '.pdf':
            return extract_text_from_pdf(file_obj)
        elif ext == '.docx':
            return extract_text_from_docx(file_obj)
        elif ext == '.txt':
            return extract_text_from_txt(file_obj)
        else:
            logging.error(f"Unsupported file type for text extraction: {ext} for file {filename}")
            return ""
    except Exception as e:
        logging.error(f"Failed to extract text from '{filename}': {e}")
        return ""


def llm_call(resume_text, job_role, experience_info, job_description_text=None):
    """
    Calls the LLM (Gemini) to analyze the resume and generate a JSON summary.
    Includes target experience criteria and optional job description in the prompt for tailored analysis.
    """
    global llm # Access the global llm instance

    if llm is None and GOOGLE_API_KEY:
        try:
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY, temperature=0.7)
            logging.info("LLM initialized successfully within llm_call.")
        except Exception as e:
            logging.error(f"Failed to initialize LLM in llm_call: {e}")
            return {"error": f"Failed to initialize LLM: {e}"}
    elif llm is None:
        logging.error("LLM not initialized. GOOGLE_API_KEY might be missing or invalid.")
        return {"error": "LLM not initialized. GOOGLE_API_KEY might be missing."}

    job_description_section = ""
    if job_description_text:
        job_description_section = f"""
**Job Description for Reference:**
{job_description_text}
"""

    min_years_for_llm = 0
    max_years_for_llm = 0
    if "at least" in experience_info:
        match = re.search(r"at least (\d+) years", experience_info)
        if match:
            min_years_for_llm = int(match.group(1))
    elif "-" in experience_info and "years" in experience_info:
        match = re.search(r"(\d+)-(\d+) years", experience_info)
        if match:
            min_years_for_llm = int(match.group(1))
            max_years_for_llm = int(match.group(2))

    # --- SIMPLIFIED PROMPT FOR FULL DATA POPULATION (NO JSON SCHEMA EXAMPLE TO AVOID PARSING ISSUES) ---
    # Removed the explicit `JSON Schema:` section and the `json_schema_string` variable.
    # We are now relying purely on the LLM's understanding of the desired structure from the textual description.
    summary_template = """
**YOUR RESPONSE MUST BE A SINGLE, VALID JSON OBJECT. NO OTHER TEXT, NO MARKDOWN FENCES (```json), NO EXPLANATIONS.**

You are an expert HR evaluator. Your primary goal is to provide a **comprehensive and complete JSON analysis** of the candidate's resume against the specified job role and requirements.

**Crucial Instruction: YOU MUST FILL ALL FIELDS IN THE JSON SCHEMA BELOW.**
If a piece of information is genuinely not found in the resume, explicitly state "Not Found" for string fields, "0/X" for scores, or an empty array `[]` for lists, but **DO NOT leave any field missing or null**.

**Salary Guidelines:**
- Less than 2 years experience: ₹15,000 to ₹25,000
- 2 to 5 years experience: ₹35,000 to ₹65,000
- More than 5 years experience: ₹50,000 to ₹80,000

**Candidate Resume to Analyze:**
{resume_text}

**Job Role:** {job_role}
**Desired Experience:** {experience_info}

{job_description_section}

**JSON Fields (REQUIRED - MUST BE PRESENT):**
- **full_name**: string (e.g., "John Doe" or "Not Found")
- **contact_number**: string (e.g., "+1-555-123-4567" or "Not Found")
- **overall_experience**: string (e.g., "5 years 3 months" or "Not Found")
- **current_company_name**: string (e.g., "Acme Corp" or "Not Found")
- **current_company_address**: string (e.g., "New York, NY" or "Not Found")
- **hiring_recommendation**: string ("Hire", "Marginally Fit", or "Reject")
- **suggested_salary_range**: string (e.g., "₹8 LPA - ₹12 LPA" or "Not Applicable (No Experience)")
- **experience_match**: string ("Good Match", "Underqualified", "Overqualified", or "No Resume Provided")
- **analysis_summary**: object containing:
    - **candidate_overview**: string (detailed summary, use `\\n` for newlines)
    - **technical_prowess**: object (e.g., {{"Languages": ["Python"], "Tools": ["Git"]}} or {{}} if none)
    - **project_impact**: array of objects (each with "title", "company", "role", "achievements": ["string"] or [])
    - **education_certifications**: object (e.g., {{"education": ["B.Sc. Computer Science"], "certifications": ["AWS Certified"]}} or {{"education": [], "certifications": []}})
    - **overall_rating**: string (e.g., "4.5/5" or "Not Rated")
    - **conclusion**: string (overall concluding remarks, use `\\n` for newlines)
- **candidate_fitment_analysis**: object containing:
    - **strategic_alignment**: string
    - **comparable_experience_analysis**: string
    - **quantifiable_impact**: string
    - **potential_gaps_risks**: string
- **scoring_matrix**: array of 5 objects (each with "dimension", "weight", "score", "justification")
    - Ensure scores are in "x/Y" format.
- **aggregate_score**: string (e.g., "90/100" or "0/100")
- **fitment_verdict**: string ("SELECTED", "MARGINALLY FIT", or "REJECTED")
- **bench_recommendation**: object containing:
    - **retain**: string ("Yes", "No", or "Maybe")
    - **ideal_future_roles**: array of strings (e.g., ["Senior Data Scientist"] or [])
    - **justification**: string
- **alternative_role_recommendations**: array of objects (each with "role", "explanation")
- **automated_recruiter_insights**: object containing:
    - **red_flag_detection**: string (e.g., "Frequent job hopping" or "None detected")
    - **growth_indicators**: string (e.g., "Consistent upskilling" or "Not evident")
    - **cross_industry_potential**: string (e.g., "Transferable skills to finance" or "Limited")
- **interview_questions**: array of 7 relevant interview questions (strings).
"""
    
    # input_variables for PromptTemplate need to reflect the *actual* placeholders in the template string.
    input_variables = ["resume_text", "job_role", "experience_info", "job_description_section"]

    summary_prompt = PromptTemplate(
        input_variables=input_variables,
        template=summary_template
    )

    chain = LLMChain(llm=llm, prompt=summary_prompt)

    try:
        logging.info("Invoking LLMChain for resume analysis.")
        result = chain.invoke(input={
            "resume_text": resume_text,
            "job_role": job_role,
            "experience_info": experience_info,
            "job_description_section": job_description_section
        })
        llm_output = result["text"].strip()
        logging.info(f"Raw LLM output received (first 500 chars): {llm_output[:500]}...")

        # --- Aggressive cleanup of LLM output before JSON parsing ---
        # Remove any leading/trailing text that might not be part of the JSON
        # This regex looks for the first '{' and the last '}'
        json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
        if json_match:
            llm_output = json_match.group(0)
            logging.info("Trimmed LLM output to valid JSON boundaries.")
        else:
            logging.warning("Could not find JSON boundaries in LLM output. Attempting raw parse.")
            # Fallback to original output if boundaries not found, hoping for the best.

        # Remove common markdown fences if they somehow persist
        llm_output = llm_output.replace('```json', '').replace('```', '').strip()
        llm_output = llm_output.replace('’', "'").replace('‘', "'").replace('“', '"').replace('”', '"')
        # ------------------------------------------------------------

        if not llm_output:
            logging.warning("LLM returned an empty response after cleanup.")
            return {"error": "LLM returned empty response. Cannot parse."}

        try:
            summary = json.loads(llm_output)
            logging.info("Successfully parsed LLM output as JSON.")
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing failed: {e}. Raw output:\n```\n{llm_output}\n```")
            # Attempt a less strict parse as a last resort
            try:
                import ast
                summary = ast.literal_eval(llm_output)
                if not isinstance(summary, dict):
                    raise ValueError("`ast.literal_eval` did not result in a dictionary.")
                logging.warning("Used ast.literal_eval for JSON parsing due to initial failure.")
            except (ValueError, SyntaxError) as e_ast:
                logging.error(f"Error decoding LLM output even with fallback: {e_ast}. Raw output:\n```\n{llm_output}\n```")
                return {"error": f"Failed to decode LLM output: {e_ast}"}
        
        return summary
    except Exception as e:
        logging.error(f"An unexpected error occurred during LLM call: {e}")
        return {"error": f"An unexpected error occurred during AI analysis: {e}"}

def analyze_resume_with_llm(resume_file_obj, job_description_file_obj, job_role, experience_type, min_years, max_years):
    """
    Orchestrates the entire resume analysis process.
    1. Extracts text from the resume and job description files.
    2. Builds a structured prompt for the LLM.
    3. Calls the LLM (Langchain) to get the structured analysis.
    4. Returns the parsed analysis data.
    """
    logging.info("Starting analyze_resume_with_llm function.")
    
    # 1. Extract text from uploaded files
    resume_text = extract_text_from_document(resume_file_obj, resume_file_obj.name)
    
    job_description_text = ""
    if job_description_file_obj:
        job_description_text = extract_text_from_document(job_description_file_obj, job_description_file_obj.name)

    if not resume_text:
        logging.warning("No text extracted from resume. LLM analysis will be limited.")
    else:
        logging.info(f"Resume text extracted. Length: {len(resume_text)} chars.")
    
    if not job_description_text:
        logging.info("No text extracted from job description.")
    else:
        logging.info(f"JD text extracted. Length: {len(job_description_text)} chars.")

    # Format experience info for the LLM prompt
    experience_info = ""
    if experience_type == "Specific Range (Years)":
        experience_info = f"{min_years}-{max_years} years"
    elif experience_type == "Minimum Years Required":
        experience_info = f"at least {min_years} years"
    else:
        experience_info = experience_type

    # 2. Call the LLM service function
    analysis_data = llm_call(
        resume_text=resume_text,
        job_role=job_role,
        experience_info=experience_info,
        job_description_text=job_description_text
    )
    
    if analysis_data:
        logging.info("LLM analysis data received successfully.")
    else:
        logging.error("LLM analysis data is None or empty after API call.")

    return analysis_data

def make_blandai_call(phone_number, candidate_name, interview_questions_list):
    """
    Initiates an outbound call using Bland.ai for an interview.
    """
    if not BLANDAI_API_KEY:
        logging.error("BLANDAI_API_KEY is not set in environment variables.")
        return {"error": "BLANDAI_API_KEY is not set."}

    if interview_questions_list:
        script_core = "I have a few questions for you. " + " ".join(interview_questions_list[:3])
    else:
        script_core = "We'd like to discuss your application in more detail."

    headers = {
        "authorization": BLANDAI_API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "phone_number": phone_number,
        "task": f"Hello {candidate_name}, this is an automated call from ATS regarding your application. Are you available for a brief automated interview? {script_core}",
        "voice_id": 9,
        "reduce_latency": True,
        "transfer_list": [],
        "answered_by_human": True,
        "first_sentence": f"Hello {candidate_name}, I am an AI interviewer calling to conduct a brief interview for the position you applied for. Are you ready to proceed?"
    }

    try:
        response = requests.post("https://api.bland.ai/v1/calls", headers=headers, json=payload)
        response.raise_for_status()
        logging.info(f"Bland.ai call initiated successfully. Response: {response.json()}")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Bland.ai API call failed: {e}")
        return {"error": f"Bland.ai call failed: {e}"}

def get_blandai_call_details(call_id):
    """
    Fetches details of a Bland.ai call by its ID.
    """
    if not BLANDAI_API_KEY:
        logging.error("BLANDAI_API_KEY is not set in environment variables.")
        return {"error": "BLANDAI_API_KEY is not set."}

    headers = {
        "authorization": BLANDAI_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(f"https://api.bland.ai/v1/calls/{call_id}", headers=headers)
        response.raise_for_status()
        logging.info(f"Bland.ai call details for ID {call_id} fetched successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Bland.ai call details for ID {call_id}: {e}")
        return {"error": f"Failed to fetch call details: {e}"}

def get_blandai_call_summary(call_id):
    """
    Fetches the summary of a Bland.ai call.
    """
    if not BLANDAI_API_KEY:
        logging.error("BLANDAI_API_KEY is not set in environment variables.")
        return {"error": "BLANDAI_API_KEY is not set."}

    headers = {
        "authorization": BLANDAI_API_KEY,
        "Content-Type": "application/json"
    }

    try:
        response = requests.get(f"https://api.bland.ai/v1/calls/{call_id}/summary", headers=headers)
        response.raise_for_status()
        logging.info(f"Bland.ai call summary for ID {call_id} fetched successfully.")
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching Bland.ai call summary for ID {call_id}: {e}")
        return {"error": f"Failed to fetch call summary: {e}"}