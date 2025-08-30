import google.generativeai as genai
import json
import logging
import os
from django.conf import settings
from google.api_core import exceptions as google_exceptions


logger = logging.getLogger(__name__)

try:
    # Ensure the API key is correctly configured.
    genai.configure(api_key=settings.GEMINI_API_KEY)
except AttributeError:
    logger.error("GEMINI_API_KEY is not configured in Django settings.")
    # You might want to raise an exception here or handle it as appropriate for your application.
    raise ValueError("GEMINI_API_KEY is not configured.")

# Configure the model with proper settings.
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
]

# Use a valid model name
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash',
    generation_config=generation_config,
    safety_settings=safety_settings,
)

def create_basic_analysis_prompt(resume_text, jd_text):
    """
    Generates a prompt for a concise, basic ATS analysis.
    """
    return f"""
    You are a basic ATS (Applicant Tracking System) assistant.
    Your task is to provide a brief, high-level analysis of a resume against a job description.
    
    The analysis should be a JSON object with the following keys:
    - `summary`: A one-paragraph summary of the candidate's fit.
    - `fitment_score`: An integer score from 1 to 100 representing the overall match.
    - `key_skills_match`: A list of 5-10 key skills from the job description that are present in the resume.
    
    ---
    Resume:
    {resume_text}
    
    ---
    Job Description:
    {jd_text}
    """

def create_advanced_analysis_prompt(resume_text, jd_text):
    """
    Generates a detailed prompt for an advanced ATS analysis.
    """
    return f"""
    You are an expert HR and recruitment analyst. Your task is to perform a detailed, multi-faceted analysis of a candidate's resume against a job description.
    Provide a comprehensive JSON object with the following structure and data:
    
    {{
      "overall_verdict": "A brief summary verdict (e.g., 'Highly Recommended for Interview', 'Potential Fit, needs further review', 'Not a good fit').",
      "hiring_recommendation": "A one-sentence recommendation.",
      "aggregate_score": "An integer score from 1-100.",
      "analysis_summary": {{
        "candidate_overview": "A brief overview of the candidate's career.",
        "technical_prowess": {{
          "strengths": ["List of key technical strengths"],
          "weaknesses": ["List of potential technical gaps"]
        }},
        "project_impact": [
          {{
            "project_name": "Name",
            "impact_summary": "Summary of quantifiable impact"
          }}
        ],
        "overall_rating": "A short paragraph explaining the overall score.",
        "conclusion": "Final thoughts on the candidate's profile."
      }},
      "scoring_matrix": [
        {{
          "category": "e.g., Technical Skills",
          "score": 85,
          "notes": "Relevant details."
        }}
      ],
      "recruiter_insights": {{
        "potential_red_flags": "A summary of any concerns or red flags."
      }},
      "interview_questions": [
        "A list of 3-5 specific questions based on the resume."
      ]
    }}
    
    ---
    Resume:
    {resume_text}
    
    ---
    Job Description:
    {jd_text}
    """

def call_gemini_api(prompt):
    """
    Sends the prompt to the Gemini API and handles the response with better error handling.
    """
    try:
        response = model.generate_content(prompt)
        
        # Check for empty response
        if not response or not response.text:
            logger.error("Gemini API returned an empty or invalid response.")
            return {'error': "AI analysis failed: The API returned no content."}
            
        response_text = response.text.strip('` \n')
        
        # Remove the 'json' prefix if present
        if response_text.startswith('json'):
            response_text = response_text[4:].strip()
            
        return json.loads(response_text)
    except google_exceptions.GoogleAPIError as e:
        logger.error(f"Google API Error: {e}")
        return {'error': f"Google API Error: {e}"}
    except genai.types.BlockedPromptException as e:
        logger.error(f"Gemini API call blocked: {e}")
        return {'error': "AI analysis failed due to safety settings."}
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON from Gemini API: {e}")
        return {'error': f"AI analysis failed: Invalid JSON response. Raw text: {response_text}"}
    except Exception as e:
        logger.error(f"Gemini API call failed: {e}")
        return {'error': f"AI analysis failed: {e}"}