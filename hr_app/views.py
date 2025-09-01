from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_protect
from datetime import datetime 
import logging
from venv import logger
#import google.generativeai as genai
import phonenumbers# myhrproject/hr_app/views.py
import docx2txt
# import fitz  # PyMuPDF for PDF
import os
import re
import pythoncom
import pytz
import win32com.client
from urllib.parse import urljoin
import spacy
from django.conf import settings
from django.shortcuts import render
import win32com.client
import pythoncom
from django.shortcuts import render
import pythoncom
from collections import defaultdict
from datetime import datetime, timedelta
import os
import json
import random
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.contrib import messages
from django.urls import reverse
from django.http import JsonResponse
import requests
from .forms import ResumeUploadForm, FinalDecisionForm, PhoneNumberForm # Your existing forms
from .models import Application, CandidateAnalysis, JobDescriptionDocument # Your existing model
from . import services # Import your services.py
from django.template.defaulttags import register # To use custom filter in template
from django.db.models import Q, Case, When, IntegerField # Corrected: Import Case, When, IntegerField
import re
# Get a logger instance for navigation tracking
import logging
navigation_logger = logging.getLogger('hr_app_navigation') #
from django.views.decorators.http import require_POST
from django.core.files.base import ContentFile
# Import for authentication
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required, user_passes_test
from .forms import CustomUserCreationForm, CustomAuthenticationForm # Import your new forms
import google.api_core.exceptions # Import for more specific API error handling
import google # <--- ADDED THIS LINE
# --- Authentication Views ---
from .services import llm_call
from django.core.files.uploadedfile import SimpleUploadedFile

# Assuming these are already defined correctly
resume_storage = FileSystemStorage(location='media/resumes')
job_description_storage = FileSystemStorage(location='media/job_descriptions')

def signup_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')  # Assuming email used as username
        password = request.POST.get('password')

        if not username or not password:
            messages.error(request, "Email and password are required.")
            return render(request, 'signup.html')

        if User.objects.filter(username=username).exists():
            messages.error(request, "User already exists.")
            return render(request, 'signup.html')

        user = User.objects.create_user(username=username, password=password)
        login(request, user)
        messages.success(request, "Account created successfully!")
        return redirect('dashboard')  # Replace with your dashboard route

    return render(request, 'signup.html')

@csrf_protect
def signin_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        if not username or not password:
            messages.error(request, "Both fields are required.")
            return render(request, 'signin.html')

        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, f"Welcome back, {user.username}!")
            return redirect('dashboard')  # Replace with your dashboard route
        else:
            messages.error(request, "Invalid credentials.")
            return render(request, 'signin.html')

    return render(request, 'signin.html')

@login_required
def signout_view(request):
    """
    Handles user logout.
    """
    logout(request)
    messages.info(request, "You have been logged out.")
    return redirect('signin') # Redirect to signin page after logout

# --- Role-based Access Control Helpers ---

def is_admin(user):
    """
    Checks if the user has 'admin' or 'superadmin' role.
    """
    return user.is_authenticated and (user.role == 'admin' or user.role == 'superadmin')

def is_superadmin(user):
    """
    Checks if the user has 'superadmin' role.
    """
    return user.is_authenticated and user.role == 'superadmin'


# --- Dashboard and existing views with login_required and role checks ---

# fs = FileSystemStorage(location='media/resumes')  # Save to 'media/resumes' folder
fs = FileSystemStorage(location='media/resumes', base_url='/media/resumes/')


resume_storage = FileSystemStorage(location=settings.RESUME_UPLOAD_ROOT, base_url=settings.RESUME_UPLOAD_URL)
job_description_storage = FileSystemStorage(location=settings.JOB_DESCRIPTION_UPLOAD_ROOT, base_url=settings.JOB_DESCRIPTION_UPLOAD_URL)


# @login_required # Ensure user is logged in to access this page
def home(request):
    # """
    # Renders the home page, which will likely be the resume analysis page by default.
    # """
    # navigation_logger.info( #
    #     f"User (ID: {request.user.id if request.user.is_authenticated else 'Anonymous'}) "
    #     f"navigated to Home Page. Path: {request.path}"
    # )
    # return redirect('/home') #
    return render(request, 'home.html')

def _parse_experience_string(experience_str):
    """
    Helper function to parse an experience string (e.g., "3 years", "5+", "10")
    into a numerical integer for comparison.
    Returns an integer or None if parsing fails.
    """
    if not isinstance(experience_str, str):
        return None # Not a string, cannot parse

    # Remove "years" or "year" and any leading/trailing whitespace
    clean_str = experience_str.lower().replace('years', '').replace('year', '').strip()

    # Handle "X+" cases (e.g., "5+", "10+")
    match_plus = re.match(r'(\d+)\+', clean_str)
    if match_plus:
        try:
            return int(match_plus.group(1)) # Return the number before the plus
        except ValueError:
            return None

    # Handle pure digit cases (e.g., "3", "5")
    if clean_str.isdigit():
        try:
            return int(clean_str)
        except ValueError:
            return None
            
    return None # Could not parse

@login_required
# def resume_analysis_view(request):
#     analysis_result = None
#     resume_url = None

#     form = ResumeUploadForm(request.POST or None, request.FILES or None)

#     if request.method == 'POST':
#         logging.info("POST request received for resume analysis.")
#         if form.is_valid():
#             logging.info("Form is valid. Processing uploaded files and form data.")
#             resume_file = form.cleaned_data['resume_pdf']
#             job_description_file = form.cleaned_data['job_description']
#             job_role = form.cleaned_data['job_role']
#             target_experience_type = form.cleaned_data['target_experience_type']
#             min_years_required = form.cleaned_data['min_years_required']
#             max_years_required = form.cleaned_data['max_years_required']

#             try:
#                 # 1. Save the resume file for PDF preview
#                 resume_filename = resume_storage.save(resume_file.name, resume_file)
#                 resume_url = request.build_absolute_uri(resume_storage.url(resume_filename))
#                 logging.info(f"Resume file '{resume_filename}' saved for preview. URL: {resume_url}")

#                 # 2. Call the main AI analysis service function.
#                 llm_response = services.analyze_resume_with_llm(
#                     resume_file_obj=resume_file,
#                     job_description_file_obj=job_description_file,
#                     job_role=job_role,
#                     experience_type=target_experience_type,
#                     min_years=min_years_required,
#                     max_years=max_years_required
#                 )

#                 if llm_response and not llm_response.get("error"):
#                     analysis_result = llm_response
                    
#                     # --- START: DATABASE SAVE LOGIC ---
#                     try:
#                         # Get the analysis_summary dictionary safely
#                         analysis_summary = analysis_result.get("analysis_summary", {})
#                         candidate_fitment_analysis = analysis_result.get("candidate_fitment_analysis", {})
                        
#                         # Prepare data for the CandidateAnalysis model.
#                         # Serialize complex data to JSON strings.
#                         candidate_data_for_db = {
#                             "resume_file_path": resume_filename,
#                             "full_name": analysis_result.get("full_name"),
#                             "job_role": job_role, # Use form data for this
#                             "phone_no": analysis_result.get("contact_number"),
#                             "hiring_recommendation": analysis_result.get("hiring_recommendation"),
#                             "suggested_salary_range": analysis_result.get("suggested_salary_range"),
#                             "interview_questions": json.dumps(analysis_result.get("interview_questions", [])),
                            
#                             # Store the entire analysis_summary as a JSON string if needed,
#                             # or remove this if all sub-components are stored separately
#                             "analysis_summary": json.dumps(analysis_summary), 
                            
#                             "experience_match": analysis_result.get("experience_match"),
#                             "overall_experience": analysis_result.get("overall_experience"),
#                             "current_company_name": analysis_result.get("current_company_name"),
#                             "current_company_address": analysis_result.get("current_company_address"),
#                             "fitment_verdict": analysis_result.get("fitment_verdict"), 
#                             "aggregate_score": analysis_result.get("aggregate_score"), 
                            
#                             # Fields extracted from candidate_fitment_analysis
#                             "strategic_alignment": candidate_fitment_analysis.get("strategic_alignment", ""),
#                             "quantifiable_impact": candidate_fitment_analysis.get("quantifiable_impact", ""),
#                             "potential_gaps_risks": candidate_fitment_analysis.get("potential_gaps_risks", ""),
#                             "comparable_experience": candidate_fitment_analysis.get("comparable_experience_analysis", ""), # Note the key name difference
                            
#                             # Other top-level complex fields that need JSON dumping
#                             "scoring_matrix_json": json.dumps(analysis_result.get("scoring_matrix", [])),
#                             "bench_recommendation_json": json.dumps(analysis_result.get("bench_recommendation", {})),
#                             "alternative_role_recommendations_json": json.dumps(analysis_result.get("alternative_role_recommendations", [])),
#                             "automated_recruiter_insights_json": json.dumps(analysis_result.get("automated_recruiter_insights", {})),

#                             # NEW: Fields extracted from analysis_summary
#                             "candidate_overview": analysis_summary.get("candidate_overview", ""),
#                             "technical_prowess_json": json.dumps(analysis_summary.get("technical_prowess", {})),
#                             "project_impact_json": json.dumps(analysis_summary.get("project_impact", [])),
#                             "education_certifications_json": json.dumps(analysis_summary.get("education_certifications", {})),
#                             "overall_rating_summary": analysis_summary.get("overall_rating", ""), # Renamed to avoid conflict
#                             "conclusion_summary": analysis_summary.get("conclusion", ""), # Renamed to avoid conflict
#                         }
                        
#                         # Use .create() to save the new object and get its automatically generated ID.
#                         candidate_obj = CandidateAnalysis.objects.create(**candidate_data_for_db)
                        
#                         # Now, update the analysis_result dictionary with the new ID
#                         analysis_result['id'] = candidate_obj.id 
                        
#                         messages.success(request, f"Analysis saved to database for {candidate_obj.full_name}.")
#                     except Exception as db_save_error:
#                         logging.warning(f"AI analysis completed, but failed to save to database: {db_save_error}")
#                         messages.warning(request, f"AI analysis completed, but failed to save to database: {db_save_error}")
#                         # Even if save fails, analysis_result is still available for the page, but without an id
#                     # --- END: DATABASE SAVE LOGIC ---
                    
#                     messages.success(request, f"AI analysis completed for {analysis_result.get('full_name', 'the candidate')}.")
#                 else:
#                     error_message = llm_response.get("error", "AI analysis failed to return a valid response.") if llm_response else "LLM response was empty or None."
#                     logging.error(f"LLM response error: {error_message}")
#                     messages.error(request, error_message)
#             except Exception as e:
#                 logging.error(f"An unexpected error occurred during the analysis process: {e}", exc_info=True)
#                 messages.error(request, f"An unexpected error occurred during analysis: {e}")
#         else:
#             logging.warning("Form is not valid. Displaying errors.")
#             messages.error(request, "Please correct the errors in the form before submitting.")

#     context = {
#         'form': form,
#         'analysis_result': analysis_result,
#         'resume_url': resume_url,
#     }

#     return render(request, 'resume_analysis.html', context)






def resume_analysis_view(request):
    analysis_result = None
    resume_url = None
    
    # Get all existing job description documents
    job_description_documents = JobDescriptionDocument.objects.all()

    form = ResumeUploadForm(request.POST or None, request.FILES or None)

    if request.method == 'POST':
        logging.info("POST request received for resume analysis.")
        
        # Check if an existing JD was selected from the dropdown
        existing_jd_id = request.POST.get('job_description_id')
        
        if form.is_valid():
            logging.info("Form is valid. Processing uploaded files and form data.")
            resume_file = form.cleaned_data['resume_pdf']
            
            # Use uploaded job description file first, if it exists
            if 'job_description' in request.FILES:
                job_description_file = form.cleaned_data['job_description']
            elif existing_jd_id:
                # If an existing JD was selected, fetch it from the database
                try:
                    job_description_doc = JobDescriptionDocument.objects.get(pk=existing_jd_id)
                    job_description_file = job_description_doc.file
                    logging.info(f"Using existing job description from database: {job_description_doc.title}")
                except JobDescriptionDocument.DoesNotExist:
                    messages.error(request, "Selected job description not found.")
                    logging.error(f"Job Description with ID {existing_jd_id} not found.")
                    job_description_file = None
            else:
                messages.error(request, "Please upload or select a job description.")
                job_description_file = None
            
            job_role = form.cleaned_data['job_role']
            target_experience_type = form.cleaned_data['target_experience_type']
            min_years_required = form.cleaned_data['min_years_required']
            max_years_required = form.cleaned_data['max_years_required']

            if job_description_file:
                try:
                    # 1. Save the resume file for PDF preview
                    resume_filename = resume_storage.save(resume_file.name, resume_file)
                    resume_url = request.build_absolute_uri(resume_storage.url(resume_filename))
                    logging.info(f"Resume file '{resume_filename}' saved for preview. URL: {resume_url}")

                    # 2. Call the main AI analysis service function.
                    llm_response = services.analyze_resume_with_llm(
                        resume_file_obj=resume_file,
                        job_description_file_obj=job_description_file,
                        job_role=job_role,
                        experience_type=target_experience_type,
                        min_years=min_years_required,
                        max_years=max_years_required
                    )

                    if llm_response and not llm_response.get("error"):
                        analysis_result = llm_response
                        
                        # --- START: DATABASE SAVE LOGIC ---
                        try:
                            # Get the analysis_summary dictionary safely
                            analysis_summary = analysis_result.get("analysis_summary", {})
                            candidate_fitment_analysis = analysis_result.get("candidate_fitment_analysis", {})
                            
                            # Prepare data for the CandidateAnalysis model.
                            # Serialize complex data to JSON strings.
                            candidate_data_for_db = {
                                "resume_file_path": resume_filename,
                                "full_name": analysis_result.get("full_name"),
                                "job_role": job_role, # Use form data for this
                                "phone_no": analysis_result.get("contact_number"),
                                "hiring_recommendation": analysis_result.get("hiring_recommendation"),
                                "suggested_salary_range": analysis_result.get("suggested_salary_range"),
                                "interview_questions": json.dumps(analysis_result.get("interview_questions", [])),
                                
                                # Store the entire analysis_summary as a JSON string if needed,
                                # or remove this if all sub-components are stored separately
                                "analysis_summary": json.dumps(analysis_summary), 
                                
                                "experience_match": analysis_result.get("experience_match"),
                                "overall_experience": analysis_result.get("overall_experience"),
                                "current_company_name": analysis_result.get("current_company_name"),
                                "current_company_address": analysis_result.get("current_company_address"),
                                "fitment_verdict": analysis_result.get("fitment_verdict"), 
                                "aggregate_score": analysis_result.get("aggregate_score"), 
                                
                                # Fields extracted from candidate_fitment_analysis
                                "strategic_alignment": candidate_fitment_analysis.get("strategic_alignment", ""),
                                "quantifiable_impact": candidate_fitment_analysis.get("quantifiable_impact", ""),
                                "potential_gaps_risks": candidate_fitment_analysis.get("potential_gaps_risks", ""),
                                "comparable_experience": candidate_fitment_analysis.get("comparable_experience_analysis", ""), # Note the key name difference
                                
                                # Other top-level complex fields that need JSON dumping
                                "scoring_matrix_json": json.dumps(analysis_result.get("scoring_matrix", [])),
                                "bench_recommendation_json": json.dumps(analysis_result.get("bench_recommendation", {})),
                                "alternative_role_recommendations_json": json.dumps(analysis_result.get("alternative_role_recommendations", [])),
                                "automated_recruiter_insights_json": json.dumps(analysis_result.get("automated_recruiter_insights", {})),

                                # NEW: Fields extracted from analysis_summary
                                "candidate_overview": analysis_summary.get("candidate_overview", ""),
                                "technical_prowess_json": json.dumps(analysis_summary.get("technical_prowess", {})),
                                "project_impact_json": json.dumps(analysis_summary.get("project_impact", [])),
                                "education_certifications_json": json.dumps(analysis_summary.get("education_certifications", {})),
                                "overall_rating_summary": analysis_summary.get("overall_rating", ""), # Renamed to avoid conflict
                                "conclusion_summary": analysis_summary.get("conclusion", ""), # Renamed to avoid conflict
                            }
                            
                            # Use .create() to save the new object and get its automatically generated ID.
                            candidate_obj = CandidateAnalysis.objects.create(**candidate_data_for_db)
                            
                            # Now, update the analysis_result dictionary with the new ID
                            analysis_result['id'] = candidate_obj.id 
                            
                            messages.success(request, f"Analysis saved to database for {candidate_obj.full_name}.")
                        except Exception as db_save_error:
                            logging.warning(f"AI analysis completed, but failed to save to database: {db_save_error}")
                            messages.warning(request, f"AI analysis completed, but failed to save to database: {db_save_error}")
                            # Even if save fails, analysis_result is still available for the page, but without an id
                        # --- END: DATABASE SAVE LOGIC ---
                        
                        messages.success(request, f"AI analysis completed for {analysis_result.get('full_name', 'the candidate')}.")
                    else:
                        error_message = llm_response.get("error", "AI analysis failed to return a valid response.") if llm_response else "LLM response was empty or None."
                        logging.error(f"LLM response error: {error_message}")
                        messages.error(request, error_message)
                except Exception as e:
                    logging.error(f"An unexpected error occurred during the analysis process: {e}", exc_info=True)
                    messages.error(request, f"An unexpected error occurred during analysis: {e}")
            
        else:
            logging.warning("Form is not valid. Displaying errors.")
            messages.error(request, "Please correct the errors in the form before submitting.")

    context = {
        'form': form,
        'analysis_result': analysis_result,
        'resume_url': resume_url,
        'job_description_documents': job_description_documents, # Pass the documents to the template
    }

    return render(request, 'resume_analysis.html', context)

@login_required
def interview_dashboard_view(request):
    """
    Displays a dashboard of all candidates with their high-level interview status.
    This will be mapped to the 'interview_status' URL name.
    """
    unique_job_roles = CandidateAnalysis.objects.values_list('job_role', flat=True).distinct().exclude(job_role__isnull=True).exclude(job_role__exact='').order_by('job_role') #

    all_candidates_query = CandidateAnalysis.objects.filter(interview_status='Pending') #
    completed_interview = CandidateAnalysis.objects.filter(interview_status='Complete')

    selected_job_role = request.GET.get('job_role') #
    if selected_job_role: #
        all_candidates_query = all_candidates_query.filter(job_role=selected_job_role) #

    try: #
        all_candidates = all_candidates_query.order_by('-created_at') #
    except Exception: # Catch any potential FieldError
        all_candidates = all_candidates_query.order_by('-id') #
        messages.warning(request, "Could not sort by 'created_at'. Sorting by creation order (ID) instead. Consider adding a 'created_at' or 'last_updated' field to your CandidateAnalysis model.") #

    for candidate in all_candidates: #
        if candidate.bland_call_id: #
            try: #
                call_details = services.get_blandai_call_details(candidate.bland_call_id) #
                if call_details and not call_details.get('error'): #
                    candidate.call_details = call_details #
                else:
                    candidate.call_details = {'status': 'error', 'error': call_details.get('error', 'API Error')} #
            except Exception as e: #
                candidate.call_details = {'status': 'error', 'error': f'Fetch failed: {e}'} #
        else:
            candidate.call_details = None #

    context = { #
        'all_candidates': all_candidates, #
        'unique_job_roles': unique_job_roles, #
        'completed_interview':completed_interview
    }
    return render(request, 'interview_dashboard.html', context) #


@login_required
def candidate_profile(request):

    return render(request, 'candidate_profile.html') #

@login_required
def interview_detail_view(request, candidate_id):
    """
    Displays detailed interview status for a single candidate and allows triggering Bland.ai calls.
    This replaces much of the old 'interview_status_view' functionality and is linked from the dashboard tiles.
    """
    candidate = get_object_or_404(CandidateAnalysis, id=candidate_id) #
    phone_form = PhoneNumberForm(initial={'phone_number': candidate.phone_no}) #

    call_details = None #
    call_summary = None #

    if candidate.bland_call_id: #
        call_details = services.get_blandai_call_details(candidate.bland_call_id) #
        if call_details and not call_details.get('error'): #
            if call_details.get('status') == 'completed': #
                call_summary = services.get_blandai_call_summary(candidate.bland_call_id) #
                if call_summary and not call_summary.get('error'): #
                    messages.info(request, "Call completed and summary fetched.") #
                else:
                    messages.warning(request, call_summary.get('error', "Could not fetch call summary.")) #
            else:
                messages.info(request, f"Call status: {call_details.get('status', 'Unknown').replace('-', ' ').title()}. Refresh to check for updates.") #
        else:
            messages.warning(request, call_details.get('error', "Could not fetch call details for the existing call ID.")) #
    
    if request.method == 'POST': #
        action = request.POST.get('action') #

        if action == 'start_call': #
            phone_form = PhoneNumberForm(request.POST) #
            if phone_form.is_valid(): #
                phone_number = phone_form.cleaned_data['phone_number'] #
                
                if not phone_number and candidate.phone_no: #
                    phone_number = candidate.phone_no #
                
                if not phone_number: #
                    messages.error(request, "A valid phone number is required to start the call. It must start with '+' and country code, e.g., +919876543210.") #
                else:
                    if candidate.interview_questions: #
                        try: #
                            call_response = services.make_blandai_call(phone_number, candidate.full_name, json.loads(candidate.interview_questions)) #
                            if call_response and not call_response.get('error'): #
                                candidate.bland_call_id = call_response.get('call_id') #
                                candidate.save() #
                                messages.success(request, f"Call initiated successfully! Call ID: {candidate.bland_call_id}. Status: {call_response.get('status', 'Initiated').replace('-', ' ').title()}") #
                                return redirect('interview_detail', candidate_id=candidate.id) #
                            else:
                                messages.error(request, call_response.get('error', "Failed to initiate call with Bland.ai.")) #
                        except json.JSONDecodeError:
                            messages.error(request, "Error parsing interview questions. Please re-analyze the resume.")
                        except Exception as e: #
                            messages.error(request, f"An error occurred while initiating call: {e}") #
                    else:
                        messages.error(request, "No interview questions generated. Please analyze a resume first.") #
            else:
                messages.error(request, "Invalid phone number provided in the form.") #
        
        elif action == 'fetch_status' and candidate and candidate.bland_call_id: #
            return redirect('interview_detail', candidate_id=candidate.id) #

    context = { #
        'candidate': candidate, #
        'call_details': call_details, #
        'call_summary': call_summary, #
        'phone_form': phone_form, #
    }
    return render(request, 'interview_detail.html', context) #


@login_required
def interview_status_view(request, candidate_id=None):
    if candidate_id:
        single_candidate = get_object_or_404(CandidateAnalysis, id=candidate_id)
        single_call_details = None
        single_call_summary = None
        
        phone_form = PhoneNumberForm(initial={'phone_number': single_candidate.phone_no})
        
        # Manually create the FinalDecisionForm and populate it
        initial_decision = single_candidate.final_decision if single_candidate.final_decision else None
        initial_salary = single_candidate.final_salary if single_candidate.final_salary else None
        
        final_decision_form = FinalDecisionForm(initial={
            'final_decision': initial_decision,
            'final_salary': initial_salary
        })

        if request.method == 'POST':
            action = request.POST.get('action')

            if action == 'start_call':
                phone_form = PhoneNumberForm(request.POST)
                if phone_form.is_valid():
                    # ... your existing logic for starting a call ...
                    pass

            elif action == 'save_decision':
                final_decision_form = FinalDecisionForm(request.POST)
                if final_decision_form.is_valid():
                    # Manually save the data to the model instance
                    single_candidate.final_decision = final_decision_form.cleaned_data['final_decision']
                    single_candidate.final_salary = final_decision_form.cleaned_data['final_salary']
                    single_candidate.save()
                    messages.success(request, "Final decision and salary updated successfully!")
                else:
                    messages.error(request, "Error updating final decision. Please check your inputs.")
            
            return redirect('interview_status', candidate_id=single_candidate.id)

        if single_candidate.bland_call_id:
            single_call_details = services.get_blandai_call_details(single_candidate.bland_call_id)
            if single_call_details and single_call_details.get('status') == 'completed':
                single_call_summary = services.get_blandai_call_summary(single_candidate.bland_call_id)

        context = {
            'candidate': single_candidate,
            'call_details': single_call_details,
            'call_summary': single_call_summary,
            'phone_form': phone_form,
            'form': final_decision_form, # Pass your form instance to the template
        }
        return render(request, 'interview_status.html', context)
    else:
        # ... (your existing dashboard logic) ...
        all_candidates = CandidateAnalysis.objects.all()
        unique_job_roles = CandidateAnalysis.objects.values_list('job_role', flat=True).distinct().order_by('job_role')
        experience_options = [('1', '1 Year'), ('2', '2 Years'), ('3', '3 Years'), ('4', '4 Years'),
                              ('5', '5+ Years'), ('10+', '10+ Years')]

        job_role_filter = request.GET.get('job_role')
        experience_filter = request.GET.get('experience')

        if job_role_filter:
            all_candidates = all_candidates.filter(job_role=job_role_filter)
        if experience_filter:
            if experience_filter.isdigit():
                all_candidates = all_candidates.filter(overall_experience=int(experience_filter))
            elif experience_filter == '5+':
                all_candidates = all_candidates.filter(overall_experience__gte=5)
            elif experience_filter == '10+':
                all_candidates = all_candidates.filter(overall_experience__gte=10)
        
        context = {
            'all_candidates': all_candidates,
            'unique_job_roles': unique_job_roles,
            'experience_options': experience_options,
            'selected_job_role': job_role_filter,
            'selected_experience': experience_filter,
        }
        return render(request, 'interview_status.html', context)
    

def candidate_profile_view(request, candidate_id):
    """
    Displays a comprehensive profile for a single candidate.
    Fetches all related data including interview details and summary if available.
    Also handles POST requests for finalizing decision and salary, and interview status.
    """
    candidate = get_object_or_404(CandidateAnalysis, id=candidate_id)
    
    call_details = None
    call_summary = None
    resume_url = None # Initialize resume_url

    # Generate resume_url if resume_file_path exists
    if hasattr(candidate, 'resume_file_path') and candidate.resume_file_path:
        try:
            resume_url = request.build_absolute_uri(resume_storage.url(candidate.resume_file_path))
        except Exception as e:
            logging.error(f"Error generating resume URL for candidate {candidate_id}: {e}")
            messages.error(request, "Could not generate resume preview URL.")


    # Handle POST request for forms on this page
    if request.method == 'POST':
        form_type = request.POST.get('form_type')
        print(f"DEBUG: POST request received. form_type: {form_type}") # Debugging line

        if form_type == 'finalize_decision_form':
            logging.info(f"POST request for finalize_decision_form received for candidate ID: {candidate_id}")
            
            final_decision = request.POST.get('final_decision')
            final_salary_str = request.POST.get('final_salary')
            interview_status = request.POST.get('interview_status') # NEW: Get interview status

            # Update candidate object fields
            candidate.final_decision = final_decision
            candidate.interview_status = interview_status # NEW: Update interview status
            
            # Convert final_salary to integer, handle potential errors
            if final_salary_str:
                try:
                    candidate.final_salary = int(final_salary_str)
                except ValueError:
                    messages.error(request, "Invalid salary amount. Please enter a valid number.")
                    logging.error(f"Invalid salary amount received for candidate {candidate_id}: {final_salary_str}")
                else: # Only save if conversion was successful
                    try:
                        candidate.save()
                        messages.success(request, f"Final decision, salary, and interview status saved successfully for {candidate.full_name}.")
                        logging.info(f"Final decision, salary, and interview status saved for candidate {candidate.id}.")
                    except Exception as e:
                        messages.error(request, f"Error saving final decision, salary, and interview status: {e}")
                        logging.error(f"Error saving final decision, salary, and interview status for candidate {candidate.id}: {e}", exc_info=True)
            else:
                candidate.final_salary = None # Set to None if empty
                try:
                    candidate.save() # Save even if salary is None
                    messages.success(request, f"Final decision and interview status saved successfully for {candidate.full_name}.")
                    logging.info(f"Final decision and interview status saved (salary cleared) for candidate {candidate.id}.")
                except Exception as e:
                    messages.error(request, f"Error saving final decision and interview status: {e}")
                    logging.error(f"Error saving final decision and interview status for candidate {candidate.id}: {e}", exc_info=True)
        
        elif form_type == 'initiate_interview_form':
            logging.info(f"POST request for initiate_interview_form received for candidate ID: {candidate_id}")
            phone_number = request.POST.get('phone_number')
            
            # Here you would add your logic to initiate the AI interview call
            # For example:
            # try:
            #     call_response = services.initiate_blandai_call(candidate_id, phone_number)
            #     if call_response and call_response.get('success'):
            #         candidate.bland_call_id = call_response.get('call_id') # Save call ID if your model has it
            #         candidate.save()
            #         messages.success(request, f"AI interview call initiated for {candidate.full_name} to {phone_number}.")
            #     else:
            #         messages.error(request, f"Failed to initiate AI interview call: {call_response.get('error', 'Unknown error')}")
            # except Exception as e:
            #     messages.error(request, f"An error occurred while initiating interview: {e}")
            
            messages.info(request, f"Initiate AI Interview form submitted for {phone_number}. (Logic to be implemented)")

        else:
            messages.warning(request, "Unknown form submitted.")
            logging.warning(f"Unknown form_type received: {form_type}. Request POST data: {request.POST}")


    # Fetch Bland AI call details if a call ID exists (this part remains the same)
    if hasattr(candidate, 'bland_call_id') and candidate.bland_call_id:
        try:
            call_details = services.get_blandai_call_details(candidate.bland_call_id)
            if call_details and call_details.get('status') == 'completed':
                call_summary = services.get_blandai_call_summary(candidate.bland_call_id)
        except Exception as e:
            logging.error(f"Error fetching BlandAI call details or summary: {e}")
            messages.error(request, f"Error fetching AI interview details: {e}")

    # Initialize lists/dicts for JSON fields to ensure they are always available in context
    interview_questions_list = []
    scoring_matrix_list = []
    bench_recommendation_dict = {}
    alternative_role_recommendations_list = []
    automated_recruiter_insights_dict = {}
    technical_prowess_dict = {}
    project_impact_list = []
    education_certifications_dict = {}
    analysis_summary_dict = {} # Keep for compatibility with core_competencies_assessment

    # Attempt to parse JSON fields from the candidate object
    if candidate.interview_questions:
        try:
            interview_questions_list = json.loads(candidate.interview_questions)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing interview questions for this candidate.")
    
    if candidate.scoring_matrix_json:
        try:
            scoring_matrix_list = json.loads(candidate.scoring_matrix_json)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing scoring matrix for this candidate.")

    if candidate.bench_recommendation_json:
        try:
            bench_recommendation_dict = json.loads(candidate.bench_recommendation_json)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing bench recommendation for this candidate.")

    if candidate.alternative_role_recommendations_json:
        try:
            alternative_role_recommendations_list = json.loads(candidate.alternative_role_recommendations_json)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing alternative role recommendations for this candidate.")

    if candidate.automated_recruiter_insights_json:
        try:
            automated_recruiter_insights_dict = json.loads(candidate.automated_recruiter_insights_json)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing automated recruiter insights for this candidate.")

    if candidate.technical_prowess_json:
        try:
            technical_prowess_dict = json.loads(candidate.technical_prowess_json)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing technical prowess for this candidate.")

    if candidate.project_impact_json:
        try:
            project_impact_list = json.loads(candidate.project_impact_json)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing project impact for this candidate.")

    if candidate.education_certifications_json:
        try:
            education_certifications_dict = json.loads(candidate.education_certifications_json)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing education and certifications for this candidate.")

    # This is kept as analysis_summary_dict.core_competencies_assessment is still used in HTML
    if candidate.analysis_summary: 
        try:
            analysis_summary_dict = json.loads(candidate.analysis_summary)
        except json.JSONDecodeError:
            messages.error(request, "Error parsing analysis summary for this candidate.")


    context = {
        'candidate': candidate,
        'call_details': call_details,
        'call_summary': call_summary,
        'interview_questions_list': interview_questions_list,
        'analysis_summary_dict': analysis_summary_dict, # Keep for compatibility with core_competencies_assessment
        'scoring_matrix_list': scoring_matrix_list,
        'bench_recommendation_dict': bench_recommendation_dict,
        'alternative_role_recommendations_list': alternative_role_recommendations_list,
        'automated_recruiter_insights_dict': automated_recruiter_insights_dict,
        'technical_prowess_dict': technical_prowess_dict,
        'project_impact_list': project_impact_list,
        'education_certifications_dict': education_certifications_dict,
        'resume_url': resume_url, # Pass resume_url to the template
    }
    return render(request, 'candidate_profile.html', context)
@login_required
def initiate_call_interview(request):
    """
    Displays interview status and allows triggering Bland.ai calls.
    """
    candidate = None #
    call_details = None #
    call_summary = None #
    phone_form = PhoneNumberForm() #

    candidate_id = request.session.get('current_candidate_id') #
    if candidate_id: #
        candidate = get_object_or_404(CandidateAnalysis, id=candidate_id) #
        
        if candidate.phone_no: #
            phone_form = PhoneNumberForm(initial={'phone_number': candidate.phone_no}) #

        if candidate.bland_call_id: #
            call_details = services.get_blandai_call_details(candidate.bland_call_id) #
            if call_details and not call_details.get('error'): #
                if call_details.get('status') == 'completed': #
                    call_summary = services.get_blandai_call_summary(candidate.bland_call_id) #
                    if call_summary and not call_summary.get('error'): #
                        messages.info(request, "Call completed and summary fetched.") #
                    else:
                        messages.warning(request, call_summary.get('error', "Could not fetch call summary.")) #
                else:
                    messages.info(request, f"Call status: {call_details.get('status').replace('-', ' ').title()}. Refresh to check for updates.") #
            else:
                messages.warning(request, call_details.get('error', "Could not fetch call details for the existing call ID.")) #
    
    if request.method == 'POST': #
        action = request.POST.get('action') #

        if action == 'start_call' and candidate: #
            phone_form = PhoneNumberForm(request.POST) #
            if phone_form.is_valid(): #
                phone_number = phone_form.cleaned_data['phone_number'] #
                
                if not phone_number and candidate.phone_no: #
                    phone_number = candidate.phone_no #
                
                if not phone_number or not phone_form.is_valid_phone_number(phone_number): #
                    messages.error(request, "A valid phone number is required to start the call. It must start with '+' and country code, e.g., +919876543210.") #
                else:
                    if candidate.interview_questions: #
                        try:
                            questions = json.loads(candidate.interview_questions)
                            call_response = services.make_blandai_call(phone_number, candidate.full_name, questions) #
                            if call_response and not call_response.get('error'): #
                                candidate.bland_call_id = call_response.get('call_id') #
                                candidate.save() #
                                messages.success(request, f"Call initiated successfully! Call ID: {candidate.bland_call_id}. Status: {call_response.get('status').replace('-', ' ').title()}") #
                                return redirect('interview_status') #
                            else:
                                messages.error(request, call_response.get('error', "Failed to initiate call with Bland.ai.")) #
                        except json.JSONDecodeError:
                            messages.error(request, "Error parsing interview questions. Please re-analyze the resume.")
                        except Exception as e: #
                            messages.error(request, f"An error occurred while initiating call: {e}") #
                    else:
                        messages.error(request, "No interview questions generated. Please analyze a resume first.") #
            else:
                messages.error(request, "Invalid phone number provided in the form.") #
        
        elif action == 'fetch_status' and candidate and candidate.bland_call_id: #
            return redirect('interview_status') #

    context = { #
        'candidate': candidate, #
        'call_details': call_details, #
        'call_summary': call_summary, #
        'phone_form': phone_form, #
    }
    return render(request, 'initiate_call_interview.html', context) #


@login_required
def top_recommendations_view(request):
    """
    Displays top recommended candidates based on job role and AI analysis.
    """
    navigation_logger.info( #
        f"User (ID: {request.user.id if request.user.is_authenticated else 'Anonymous'}) "
        f"navigated to Top Recommendations Page. Path: {request.path}"
    )
    unique_job_roles = CandidateAnalysis.objects.values_list('job_role', flat=True).distinct().exclude(job_role__isnull=True).exclude(job_role__exact='').order_by('job_role')
    
    recommended_candidates_query = CandidateAnalysis.objects.all()

    selected_job_role = request.GET.get('job_role')
    if selected_job_role:
        recommended_candidates_query = recommended_candidates_query.filter(job_role=selected_job_role)
        messages.info(request, f"Showing recommendations for job role: {selected_job_role}")
        navigation_logger.info(f"Filtered recommendations by job role: {selected_job_role}") #
    else:
        messages.info(request, "Showing top candidates across all job roles.")
        navigation_logger.info("Viewing all recommendations (no job role filter).") #

    # Apply AI-driven recommendation logic:
    # Prioritize 'Hire' recommendations first, then 'Good Match' for experience.
    # You can customize this logic based on how your AI populates these fields.
    recommended_candidates = recommended_candidates_query.order_by(
        # Candidates recommended for hiring first
        Case( #
            When(hiring_recommendation='Hire', then=0), #
            When(hiring_recommendation='Resign', then=1), #
            When(hiring_recommendation='Reject', then=2), #
            default=3, #
            output_field=IntegerField(), #
        ),
        # Then by experience match
        Case( #
            When(experience_match='Good Match', then=0), #
            When(experience_match='Overqualified', then=1), #
            When(experience_match='Underqualified', then=2), #
            default=3, #
            output_field=IntegerField(), #
        ),
        '-created_at' # Finally, by most recent analysis
    )

    context = {
        'recommended_candidates': recommended_candidates,
        'unique_job_roles': unique_job_roles,
        'selected_job_role': selected_job_role,
        'page_heading': 'Top Recommended Candidates'
    }
    return render(request, 'top_recommendations.html', context)


@login_required
def candidate_records_view(request):
    """
    Displays all stored candidate analysis records.
    Allows for final decision and salary updates.
    """
    candidates = CandidateAnalysis.objects.all().order_by('-created_at')  #
    job_roles = CandidateAnalysis.objects.values_list('job_role', flat=True).distinct().order_by('job_role')

    final_decision_forms = {} #

    for candidate in candidates: #
        initial_data = { #
            'final_decision': candidate.final_decision, #
            'final_salary': candidate.final_salary #
        }
        final_decision_forms[candidate.id] = FinalDecisionForm(prefix=f'decision_{candidate.id}', initial=initial_data) #

    if request.method == 'POST': #
        candidate_id = request.POST.get('candidate_id') #
        if candidate_id: #
            candidate = get_object_or_404(CandidateAnalysis, id=candidate_id) #
            form = FinalDecisionForm(request.POST, prefix=f'decision_{candidate.id}') #
            
            if form.is_valid(): #
                candidate.final_decision = form.cleaned_data['final_decision'] #
                candidate.final_salary = form.cleaned_data['final_salary'] #
                candidate.save() #
                messages.success(request, f"Final decision updated for {candidate.full_name}.") #
                return redirect('records') #
            else:
                final_decision_forms[candidate.id] = form #
                messages.error(request, "Error updating final decision.") #
        else:
            messages.error(request, "Invalid request to update record.") #

    context = { #
        'candidates': candidates,  #
        'job_roles': job_roles,
        'final_decision_forms': final_decision_forms, #
    }
    return render(request, 'records.html', context) #

def selected_candidate(request):
    """
    Displays all stored candidate analysis records.
    Allows for final decision and salary updates.
    """
    # candidates = CandidateAnalysis.objects.all().order_by('-created_at') #
    candidates = CandidateAnalysis.objects.filter(final_decision='selected').order_by('-created_at')

    final_decision_forms = {} #

    for candidate in candidates: #
        initial_data = { #
            'final_decision': candidate.final_decision, #
            'final_salary': candidate.final_salary #
        }
        final_decision_forms[candidate.id] = FinalDecisionForm(prefix=f'decision_{candidate.id}', initial=initial_data) #

    if request.method == 'POST': #
        candidate_id = request.POST.get('candidate_id') #
        if candidate_id: #
            candidate = get_object_or_404(CandidateAnalysis, id=candidate_id) #
            form = FinalDecisionForm(request.POST, prefix=f'decision_{candidate.id}') #
            
            if form.is_valid(): #
                candidate.final_decision = form.cleaned_data['final_decision'] #
                candidate.final_salary = form.cleaned_data['final_salary'] #
                candidate.save() #
                messages.success(request, f"Final decision updated for {candidate.full_name}.") #
                return redirect('records') #
            else:
                final_decision_forms[candidate.id] = form #
                messages.error(request, "Error updating final decision.") #
        else:
            messages.error(request, "Invalid request to update record.") #

    context = { #
        'candidates': candidates, #
        'final_decision_forms': final_decision_forms, #
    }
    return render(request, 'selected_candidate.html', context) #

def rejected_candidate(request):
    """
    Displays all stored candidate analysis records.
    Allows for final decision and salary updates.
    """
    candidates = CandidateAnalysis.objects.filter(final_decision='Not Selected').order_by('-created_at')
    final_decision_forms = {} #

    for candidate in candidates: #
        initial_data = { #
            'final_decision': candidate.final_decision, #
            'final_salary': candidate.final_salary #
        }
        final_decision_forms[candidate.id] = FinalDecisionForm(prefix=f'decision_{candidate.id}', initial=initial_data) #

    if request.method == 'POST': #
        candidate_id = request.POST.get('candidate_id') #
        if candidate_id: #
            candidate = get_object_or_404(CandidateAnalysis, id=candidate_id) #
            form = FinalDecisionForm(request.POST, prefix=f'decision_{candidate.id}') #
            
            if form.is_valid(): #
                candidate.final_decision = form.cleaned_data['final_decision'] #
                candidate.final_salary = form.cleaned_data['final_salary'] #
                candidate.save() #
                messages.success(request, f"Final decision updated for {candidate.full_name}.") #
                return redirect('records') #
            else:
                final_decision_forms[candidate.id] = form #
                messages.error(request, "Error updating final decision.") #
        else:
            messages.error(request, "Invalid request to update record.") #

    context = { #
        'candidates': candidates, #
        'final_decision_forms': final_decision_forms, #
    }
    return render(request, 'rejected_candidate.html', context) #

def shortlisted_candidate(request):
    """
    Displays all stored candidate analysis records.
    Allows for final decision and salary updates.
    """
    candidates = CandidateAnalysis.objects.filter(final_decision='shortlisted').order_by('-created_at')
    final_decision_forms = {} #

    for candidate in candidates: #
        initial_data = { #
            'final_decision': candidate.final_decision, #
            'final_salary': candidate.final_salary #
        }
        final_decision_forms[candidate.id] = FinalDecisionForm(prefix=f'decision_{candidate.id}', initial=initial_data) #

    if request.method == 'POST': #
        candidate_id = request.POST.get('candidate_id') #
        if candidate_id: #
            candidate = get_object_or_404(CandidateAnalysis, id=candidate_id) #
            form = FinalDecisionForm(request.POST, prefix=f'decision_{candidate.id}') #
            
            if form.is_valid(): #
                candidate.final_decision = form.cleaned_data['final_decision'] #
                candidate.final_salary = form.cleaned_data['final_salary'] #
                candidate.save() #
                messages.success(request, f"Final decision updated for {candidate.full_name}.") #
                return redirect('records') #
            else:
                final_decision_forms[candidate.id] = form #
                messages.error(request, "Error updating final decision.") #
        else:
            messages.error(request, "Invalid request to update record.") #

    context = { #
        'candidates': candidates, #
        'final_decision_forms': final_decision_forms, #
    }
    return render(request, 'shortlisted_candidate.html', context) #

@login_required
def airtable_data_view(request):
    """
    Fetches and displays all data from the specified Airtable table.
    """
    records_df = services.fetch_airtable_data(table_name="Candidates") #
    
    if isinstance(records_df, dict) and records_df.get('error'): #
        messages.error(request, records_df.get('error')) #
        records_html = "<p>Error fetching Airtable data.</p>" #
    elif not records_df.empty: #
        records_html = records_df.to_html(classes="table table-striped table-bordered", index=False) #
    else:
        records_html = "<p>No records found in Airtable.</p>" #

    context = { #
        'records_html': records_html #
    }
    return render(request, 'hr_app/airtable_data.html', context) #

@login_required
def post_data_to_airtable_view(request):
    """
    Dynamically creates a form based on Airtable schema and allows posting data.
    """
    table_name = "Candidates" #
    schema_fields = services.get_airtable_schema(table_name) #
    
    if isinstance(schema_fields, dict) and schema_fields.get('error'): #
        messages.error(request, schema_fields.get('error')) #
        return render(request, 'hr_app/post_airtable.html', {'form': None, 'messages': messages.get_messages(request)}) #

    from django import forms as django_forms #
    
    DynamicAirtableForm = type('DynamicAirtableForm', (django_forms.Form,), {}) #

    for field in schema_fields: #
        field_name = field.get('name') #
        field_type = field.get('type') #
        if field_name: #
            if field_type == 'singleLineText' or field_type == 'multilineText': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.CharField( #
                    label=field_name.replace('_', ' ').title(), #
                    required=field.get('options', {}).get('isRequired', False), #
                    max_length=255 if field_type == 'singleLineText' else None, #
                    widget=django_forms.TextInput if field_type == 'singleLineText' else django_forms.Textarea #
                )
            elif field_type == 'number': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.DecimalField( #
                    label=field_name.replace('_', ' ').title(), #
                    required=field.get('options', {}).get('isRequired', False) #
                )
            elif field_type == 'checkbox': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.BooleanField( #
                    label=field_name.replace('_', ' ').title(), #
                    required=field.get('options', {}).get('isRequired', False), #
                    initial=False #
                )
            elif field_type == 'date': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.DateField( #
                    label=field_name.replace('_', ' ').title(), #
                    required=field.get('options', {}).get('isRequired', False), #
                    widget=django_forms.DateInput(attrs={'type': 'date'}) #
                )
            elif field_type == 'dateTime': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.DateTimeField( #
                    label=field_name.replace('_', ' ').title(), #
                    required=field.get('options', {}).get('isRequired', False), #
                    widget=django_forms.DateTimeInput(attrs={'type': 'datetime-local'}) #
                )
            elif field_type == 'email': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.EmailField( #
                    label=field_name.replace('_', ' ').title(), #
                    required=field.get('options', {}).get('isRequired', False) #
                )
            elif field_type == 'url': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.URLField( #
                    label=field_name.replace('_', ' ').title(), #
                    required=field.get('options', {}).get('isRequired', False) #
                )
            elif field_type == 'singleSelect': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.CharField( #
                    label=field_name.replace('_', ' ').title() + " (Single Select)", #
                    required=field.get('options', {}).get('isRequired', False), #
                    max_length=255 #
                )
            elif field_type == 'multipleSelect': #
                DynamicAirtableForm.base_fields[field_name] = django_forms.CharField( #
                    label=field_name.replace('_', ' ').title() + " (Comma-separated Multi Select)", #
                    required=field.get('options', {}).get('isRequired', False), #
                    help_text="Enter comma-separated values for multi-select fields." #
                )
            else:
                DynamicAirtableForm.base_fields[field_name] = django_forms.CharField( #
                    label=field_name.replace('_', ' ').title() + f" (Type: {field_type})", #
                    required=field.get('options', {}).get('isRequired', False), #
                    help_text=f"Field type '{field_type}' not explicitly handled. Enter as text." #
                )

    form = DynamicAirtableForm() #

    if request.method == 'POST': #
        form = DynamicAirtableForm(request.POST) #
        if form.is_valid(): #
            data_to_post = form.cleaned_data #
            
            for field in schema_fields: #
                if field.get('type') == 'multipleSelect' and field.get('name') in data_to_post: #
                    if isinstance(data_to_post[field.get('name')], str): #
                        data_to_post[field.get('name')] = [s.strip() for s in data_to_post[field.get('name')].split(',') if s.strip()] #
            
            final_data_for_airtable = {k: v for k, v in data_to_post.items() if v is not None and v != ''} #

            post_response = services.post_data_to_airtable(table_name, final_data_for_airtable) #
            if post_response and not post_response.get('error'): #
                messages.success(request, "Data successfully posted to Airtable!") #
                form = DynamicAirtableForm() #
            else:
                messages.error(request, post_response.get('error', "Failed to post data to Airtable.")) #
        else:
            messages.error(request, "Please correct the errors in the form.") #

    context = { #
        'form': form, #
        'table_name': table_name, #
    }
    return render(request, 'post_airtable.html', context) #
    
try: #
    import win32com.client #
    WIN32COM_AVAILABLE = True #
except ImportError: #
    WIN32COM_AVAILABLE = False #
    print("Warning: 'pywin32' library not found. Local Outlook access will not work.") #
    print("Please install it using: pip install pywin32") #



# def show_applications_view(request):
#     return render(request, 'show_application.html')







# -------------------------
# Django view (remains unchanged)
# -------------------------


GEMINI_API_KEY = "AIzaSyBgZUbdu3hIwP5hmOkwtgVKKFNtLXx9j0U"
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# def show_unread_emails(request):
#     """
#     Connects to Outlook, downloads resume attachments, analyzes them with Gemini,
#     saves the extracted information to the database, and marks emails as read.
#     This version uses the real Gemini API to extract data from resumes.
#     """
#     attachments_folder = os.path.join(settings.BASE_DIR, 'media', 'resumes')

#     # Ensure the attachments folder exists
#     if not os.path.exists(attachments_folder):
#         os.makedirs(attachments_folder)

#     try:
#         # Initialize COM library and connect to Outlook
#         pythoncom.CoInitialize()
#         outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
#         inbox = outlook.GetDefaultFolder(6)  # 6 refers to the inbox folder
#         messages = inbox.Items

#         # Iterate through messages to find and process unread emails
#         for message in messages:
#             # Check for unread emails with attachments
#             if message.UnRead and message.Attachments.Count > 0:
#                 for attachment in message.Attachments:
#                     # Get the original file name and extension
#                     original_file_name = attachment.FileName
#                     name, extension = os.path.splitext(original_file_name)
                    
#                     # Create a unique filename by appending a timestamp
#                     timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
#                     unique_file_name = f"{name}_{timestamp}{extension}"

#                     file_path = os.path.join(attachments_folder, unique_file_name)
                    
#                     # Construct a URL for the database lookup using the unique name
#                     resume_url_for_db = f"/media/resumes/{unique_file_name}"

#                     try:
#                         attachment.SaveAsFile(file_path)
#                         logger.info(f"Downloaded attachment: {unique_file_name}")

#                         # Read the content of the saved document
#                         file_content = ""
#                         if file_path.endswith('.pdf'):
#                             try:
#                                 doc = fitz.open(file_path)
#                                 text = ""
#                                 for page in doc:
#                                     text += page.get_text()
#                                 file_content = text
#                                 doc.close()
#                             except Exception as e:
#                                 logger.error(f"Error reading PDF file {file_path}: {e}")
#                                 continue
#                         elif file_path.endswith('.docx'):
#                             try:
#                                 file_content = docx2txt.process(file_path)
#                             except Exception as e:
#                                 logger.error(f"Error reading DOCX file {file_path}: {e}")
#                                 continue
#                         else:
#                             logger.warning(f"Unsupported attachment type: {original_file_name}. Skipping.")
#                             continue # Skip unsupported file types

#                         # Construct the prompt for Gemini
#                         prompt = f"""
#                         You are an expert HR assistant. Your task is to extract specific information from the following resume text.
#                         The resume text is from a job application. Your response should be a JSON object containing the following keys:
#                         'candidate_name', 'experience', 'mobile_number', 'location', 'email_address'.
                        
#                         Here is the resume text:
#                         ---
#                         {file_content}
#                         ---
                        
#                         Instructions for each field:
#                         - 'candidate_name': The full name of the candidate.
#                         - 'experience': The total years of professional experience, as an integer. If not found, use 0.
#                         - 'mobile_number': The candidate's mobile number, including the country code. If not found, use an empty string.
#                         - 'location': The candidate's city and country of residence.
#                         - 'email_address': The candidate's email address found within the resume.
#                         """

#                         try:
#                             # --- LIVE GEMINI API CALL ---
#                             response = model.generate_content(prompt)
#                             response_text = response.text.strip('` \n')
#                             if response_text.startswith('json'):
#                                 response_text = response_text[4:].strip()
#                             extracted_data = json.loads(response_text)
#                             # --- END OF LIVE GEMINI API CALL ---

#                             # --- FIX FOR DATETIME ERROR STARTS HERE ---
#                             sent_on_datetime = message.SentOn
#                             if sent_on_datetime.tzinfo is None or sent_on_datetime.tzinfo.utcoffset(sent_on_datetime) is None:
#                                 local_tz = pytz.timezone(settings.TIME_ZONE)
#                                 sent_on_local = local_tz.localize(sent_on_datetime)
#                                 sent_on_utc = sent_on_local.astimezone(pytz.utc)
#                             else:
#                                 sent_on_utc = sent_on_datetime.astimezone(pytz.utc)
#                             # --- FIX FOR DATETIME ERROR ENDS HERE ---


#                             # Create and save a new Application instance
#                             application = Application.objects.create(
#                                 candidate_name=extracted_data.get('candidate_name'),
#                                 from_email=message.SenderEmailAddress,
#                                 delivery_date=sent_on_utc,
#                                 experience=extracted_data.get('experience'),
#                                 mobile_number=extracted_data.get('mobile_number'),
#                                 location=extracted_data.get('location'),
#                                 email_address=extracted_data.get('email_address'),
#                                 subject=message.Subject,
#                                 resume_url=resume_url_for_db # Use the newly generated unique URL
#                             )
#                             logger.info(f"Saved application for {application.candidate_name} (ID: {application.id})")

#                             message.UnRead = False
#                             message.Save() # Mark the email as read after successful processing and saving

#                         except Exception as e:
#                             logger.error(f"Error processing document with Gemini or saving to DB for {unique_file_name}: {e}")
#                             message.UnRead = False # Still mark as read to prevent infinite loop
#                             message.Save()
#                     except Exception as e:
#                         logger.error(f"Error saving attachment {unique_file_name}: {e}")
#                         message.UnRead = False
#                         message.Save()


#     except Exception as e:
#         logger.error(f"Error connecting to Outlook or iterating messages: {e}")
#         if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#             return JsonResponse({'error': 'Failed to connect to Outlook or process emails. Please ensure it is open and configured correctly.'}, status=500)

#     # After processing, retrieve all applications to display
#     applications = Application.objects.all().order_by('-delivery_date')
    
#     # Prepare data for JSON response (for AJAX)
#     applications_data = []
#     for app in applications:
#         applications_data.append({
#             'id': app.id,
#             'candidate_name': app.candidate_name,
#             'from': app.from_email,
#             'date': app.delivery_date.isoformat() if app.delivery_date else None,
#             'experience': app.experience,
#             'mobile_number': app.mobile_number,
#             'location': app.location,
#             'email_address': app.email_address,
#             'resume_url': app.resume_url,
#             'subject': app.subject,
#         })
    
#     # Render the page or return JSON based on request type
#     if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
#         return JsonResponse({'emails': applications_data})
#     else:
#         context = {
#             'emails': applications 
#         }
#         return render(request, 'show_application.html', context)


def show_unread_emails(request):
    """
    Connects to Outlook, downloads resume attachments, analyzes them with Gemini,
    saves the extracted information to the database, and marks emails as read.
    This version uses the real Gemini API to extract data from resumes.
    """
    attachments_folder = os.path.join(settings.BASE_DIR, 'media', 'resumes')

    # Ensure the attachments folder exists
    if not os.path.exists(attachments_folder):
        os.makedirs(attachments_folder)

    try:
        # Initialize COM library and connect to Outlook
        pythoncom.CoInitialize()
        outlook = win32com.client.Dispatch("Outlook.Application").GetNamespace("MAPI")
        inbox = outlook.GetDefaultFolder(6)  # 6 refers to the inbox folder
        messages = inbox.Items

        # Iterate through messages to find and process unread emails
        for message in messages:
            # Check for unread emails with attachments
            if message.UnRead and message.Attachments.Count > 0:
                for attachment in message.Attachments:
                    # Get the original file name and extension
                    original_file_name = attachment.FileName
                    name, extension = os.path.splitext(original_file_name)
                    
                    # Create a unique filename by appending a timestamp
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S%f')
                    unique_file_name = f"{name}_{timestamp}{extension}"

                    file_path = os.path.join(attachments_folder, unique_file_name)
                    
                    # Construct a URL for the database lookup using the unique name
                    resume_url_for_db = f"/media/resumes/{unique_file_name}"

                    try:
                        attachment.SaveAsFile(file_path)
                        logger.info(f"Downloaded attachment: {unique_file_name}")

                        # Read the content of the saved document
                        file_content = ""
                        if file_path.endswith('.pdf'):
                            try:
                                doc = fitz.open(file_path)
                                text = ""
                                for page in doc:
                                    text += page.get_text()
                                file_content = text
                                doc.close()
                            except Exception as e:
                                logger.error(f"Error reading PDF file {file_path}: {e}")
                                continue
                        elif file_path.endswith('.docx'):
                            try:
                                file_content = docx2txt.process(file_path)
                            except Exception as e:
                                logger.error(f"Error reading DOCX file {file_path}: {e}")
                                continue
                        else:
                            logger.warning(f"Unsupported attachment type: {original_file_name}. Skipping.")
                            continue # Skip unsupported file types

                        # Construct the prompt for Gemini
                        prompt = f"""
                        You are an expert HR assistant. Your task is to extract specific information from the following resume text.
                        The resume text is from a job application. Your response should be a JSON object containing the following keys:
                        'candidate_name', 'experience', 'mobile_number', 'location', 'email_address'.
                        
                        Here is the resume text:
                        ---
                        {file_content}
                        ---
                        
                        Instructions for each field:
                        - 'candidate_name': The full name of the candidate.
                        - 'experience': The total years of professional experience, as an integer. If not found, use 0.
                        - 'mobile_number': The candidate's mobile number, including the country code. If not found, use an empty string.
                        - 'location': The candidate's city and country of residence.
                        - 'email_address': The candidate's email address found within the resume.
                        """

                        try:
                            # --- LIVE GEMINI API CALL ---
                            response = model.generate_content(prompt)
                            response_text = response.text.strip('` \n')
                            if response_text.startswith('json'):
                                response_text = response_text[4:].strip()
                            extracted_data = json.loads(response_text)
                            # --- END OF LIVE GEMINI API CALL ---

                            # --- FIX FOR DATETIME ERROR STARTS HERE ---
                            sent_on_datetime = message.SentOn
                            if sent_on_datetime.tzinfo is None or sent_on_datetime.tzinfo.utcoffset(sent_on_datetime) is None:
                                local_tz = pytz.timezone(settings.TIME_ZONE)
                                sent_on_local = local_tz.localize(sent_on_datetime)
                                sent_on_utc = sent_on_local.astimezone(pytz.utc)
                            else:
                                sent_on_utc = sent_on_datetime.astimezone(pytz.utc)
                            # --- FIX FOR DATETIME ERROR ENDS HERE ---


                            # Create and save a new Application instance
                            application = Application.objects.create(
                                candidate_name=extracted_data.get('candidate_name'),
                                from_email=message.SenderEmailAddress,
                                delivery_date=sent_on_utc,
                                experience=extracted_data.get('experience'),
                                mobile_number=extracted_data.get('mobile_number'),
                                location=extracted_data.get('location'),
                                email_address=extracted_data.get('email_address'),
                                subject=message.Subject,
                                resume_url=resume_url_for_db # Use the newly generated unique URL
                            )
                            logger.info(f"Saved application for {application.candidate_name} (ID: {application.id})")

                            message.UnRead = False
                            message.Save() # Mark the email as read after successful processing and saving

                        except Exception as e:
                            logger.error(f"Error processing document with Gemini or saving to DB for {unique_file_name}: {e}")
                            message.UnRead = False # Still mark as read to prevent infinite loop
                            message.Save()
                    except Exception as e:
                        logger.error(f"Error saving attachment {unique_file_name}: {e}")
                        message.UnRead = False
                        message.Save()


    except Exception as e:
        logger.error(f"Error connecting to Outlook or iterating messages: {e}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return JsonResponse({'error': 'Failed to connect to Outlook or process emails. Please ensure it is open and configured correctly.'}, status=500)

    # After processing, retrieve all applications and job descriptions to display
    applications = Application.objects.all().order_by('-delivery_date')
    job_descriptions = JobDescriptionDocument.objects.all().order_by('-uploaded_at')
    
    # Prepare data for JSON response (for AJAX)
    applications_data = []
    for app in applications:
        applications_data.append({
            'id': app.id,
            'candidate_name': app.candidate_name,
            'from': app.from_email,
            'date': app.delivery_date.isoformat() if app.delivery_date else None,
            'experience': app.experience,
            'mobile_number': app.mobile_number,
            'location': app.location,
            'email_address': app.email_address,
            'resume_url': app.resume_url,
            'subject': app.subject,
        })
    
    # Render the page or return JSON based on request type
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'emails': applications_data})
    else:
        context = {
            'emails': applications,
            'job_descriptions': job_descriptions # Add the job descriptions to the context
        }
        return render(request, 'show_application.html', context)


# In your views.py, update the analyze_resume function
def analyze_resume(request, email_id, analysis_type, job_description_id=None):
    """
    This function analyzes the resume content and returns a JSON response.
    This is where your logic for parsing and AI calls lives.
    """
    try:
        application = Application.objects.get(id=email_id)

        # Determine the resume file path
        if not application.resume_url:
            return JsonResponse({'error': 'Resume file not found for this application.'}, status=404)

        resume_file_path = os.path.join(settings.BASE_DIR, application.resume_url.lstrip('/'))

        # Read the resume content
        resume_content = ""
        if resume_file_path.endswith('.pdf'):
            try:
                doc = fitz.open(resume_file_path)
                resume_content = "".join(page.get_text() for page in doc)
                doc.close()
            except Exception as e:
                logger.error(f"Error reading PDF resume: {e}")
                return JsonResponse({'error': f"Error reading PDF resume: {e}"}, status=500)
        elif resume_file_path.endswith('.docx'):
            try:
                resume_content = docx2txt.process(resume_file_path)
            except Exception as e:
                logger.error(f"Error reading DOCX resume: {e}")
                return JsonResponse({'error': f"Error reading DOCX resume: {e}"}, status=500)

        # Get the Job Description content if an ID is provided
        job_description_content = ""
        job_description_doc = None
        if job_description_id:
            try:
                job_description_doc = JobDescriptionDocument.objects.get(id=job_description_id)
                jd_file_path = os.path.join(settings.MEDIA_ROOT, 'job_descriptions', os.path.basename(job_description_doc.file.name))

                if jd_file_path.endswith('.pdf'):
                    try:
                        doc = fitz.open(jd_file_path)
                        job_description_content = "".join(page.get_text() for page in doc)
                        doc.close()
                    except Exception as e:
                        logger.error(f"Error reading PDF job description: {e}")
                elif jd_file_path.endswith('.docx'):
                    try:
                        job_description_content = docx2txt.process(jd_file_path)
                    except Exception as e:
                        logger.error(f"Error reading DOCX job description: {e}")
            except JobDescriptionDocument.DoesNotExist:
                logger.warning(f"Job Description with ID {job_description_id} not found.")

        # Determine job role and experience information
        job_role = job_description_doc.title if job_description_id else 'General'
        experience_info = "Not specified" # This should be dynamically parsed from JD for a real app

        # Call the new service function for both analysis types
        analysis_data = llm_call(
            resume_text=resume_content,
            job_role=job_role,
            experience_info=experience_info,
            job_description_text=job_description_content
        )

        # Check if llm_call returned an error dictionary
        if "error" in analysis_data:
            return JsonResponse(analysis_data, status=500)

        # Create a new CandidateAnalysis object to store the results
        new_analysis = CandidateAnalysis.objects.create(
            full_name=application.candidate_name,
            job_role=job_role,
            resume_file_path=application.resume_url,
            analysis_summary=json.dumps(analysis_data)
        )

        # Redirect to the analysis results page
        return redirect('analysis_results', analysis_id=new_analysis.id)

    except Application.DoesNotExist:
        return JsonResponse({'error': 'Application not found.'}, status=404)
    except Exception as e:
        logger.error(f"Error analyzing resume: {e}")
        return JsonResponse({'error': f"An error occurred during analysis: {e}"}, status=500)

# def show_unread_emails(request):
#     return render(request, 'show_application.html')

@login_required
def process_ats_option(request, email_id, ats_type):
    """
    Handles the selection of Basic ATS or Premium ATS for a given email.
    This is a placeholder for actual ATS processing logic.
    """
    # In a real application, you would:
    # 1. Retrieve the email/application details based on email_id.
    # 2. Trigger the appropriate ATS analysis (Basic or Premium).
    # 3. Store the results in your database (e.g., CandidateAnalysis model).
    # 4. Redirect to a relevant page (e.g., candidate profile or an analysis report).

    if ats_type == 'basic':
        messages.success(request, f"Basic ATS processing initiated for email ID: {email_id}. (Placeholder)")
        logging.info(f"Basic ATS triggered for email ID: {email_id}")
    elif ats_type == 'premium':
        messages.info(request, f"Premium ATS processing initiated for email ID: {email_id}. (Placeholder)")
        logging.info(f"Premium ATS triggered for email ID: {email_id}")
    else:
        messages.error(request, f"Invalid ATS type: {ats_type}")
        logging.warning(f"Invalid ATS type received: {ats_type} for email ID: {email_id}")
    
    # Redirect back to the applications list or a confirmation page
    return redirect('show_applications')



@login_required
def dashboard(request):
    """
    Renders the HR Talent Scout Dashboard with various analytics and metrics
    derived from CandidateAnalysis data, dynamically fetching call details.
    """
    # 1. Fetch all candidates
    all_candidates = CandidateAnalysis.objects.all()

    # 2. Aggregate Data for Metrics and Charts by iterating through candidates
    # This is necessary because call_details is not a direct model field for filtering
    total_candidates = 0
    completed_calls = 0
    in_progress_calls = 0
    failed_calls = 0
    not_initiated_calls = 0

    job_role_counts = defaultdict(int)
    interview_status_counts = defaultdict(int)
    hiring_recommendation_counts = defaultdict(int)
    experience_buckets = defaultdict(int)

    for candidate in all_candidates:
        total_candidates += 1
        
        # Aggregate Job Role
        if candidate.job_role:
            job_role_counts[candidate.job_role] += 1

        # Aggregate Experience Level - NOW USING THE HELPER FUNCTION
        raw_exp_str = candidate.overall_experience
        exp = _parse_experience_string(raw_exp_str) # Parse the string to an integer
        
        if exp is not None: # Ensure parsed experience is not null
            if exp <= 1:
                experience_buckets['1 Year'] += 1
            elif exp == 2:
                experience_buckets['2 Years'] += 1
            elif exp == 3:
                experience_buckets['3 Years'] += 1
            elif exp == 4:
                experience_buckets['4 Years'] += 1
            elif 5 <= exp < 10: # Experience between 5 and 9 years
                experience_buckets['5-9 Years'] += 1
            else: # Experience 10 years or more (exp >= 10)
                experience_buckets['10+ Years'] += 1

        # Aggregate Interview Status and Call Metrics
        if candidate.bland_call_id:
            # Dynamically fetch call details for each candidate with a call ID
            # IMPORTANT: This might be slow if you have many candidates.
            # Consider caching or fetching in bulk if your Bland.ai API allows.
            call_details = services.get_blandai_call_details(candidate.bland_call_id)
            if call_details and not call_details.get('error'):
                status = call_details.get('status')
                if status == 'completed':
                    completed_calls += 1
                    interview_status_counts['completed'] += 1
                elif status == 'in_progress':
                    in_progress_calls += 1
                    interview_status_counts['in_progress'] += 1
                elif status == 'failed':
                    failed_calls += 1
                    interview_status_counts['failed'] += 1
                else:
                    # Catch any other statuses (e.g., 'queued', 'ringing')
                    interview_status_counts[status] += 1
            else:
                # If call_details fetch failed, treat as not initiated or failed to fetch
                interview_status_counts['failed_to_fetch'] += 1 # New category for dashboard
        else:
            not_initiated_calls += 1
            interview_status_counts['not_initiated'] += 1

        # Aggregate Hiring Recommendation
        if candidate.hiring_recommendation:
            hiring_recommendation_counts[candidate.hiring_recommendation] += 1

    # Calculate Call Success Rate
    total_calls_attempted = completed_calls + in_progress_calls + failed_calls
    success_rate = (completed_calls / total_calls_attempted * 100) if total_calls_attempted > 0 else 0
    success_rate = round(success_rate, 2)

    # 3. Prepare Chart Data Structures

    # Chart 1: Candidates by Job Role
    job_role_chart_data = {
        'labels': list(job_role_counts.keys()),
        'data': list(job_role_counts.values()),
    }

    # Chart 2: Interview Status Distribution
    interview_status_labels = []
    interview_status_data = []
    ordered_statuses = ['completed', 'in_progress', 'failed', 'not_initiated', 'failed_to_fetch'] # Include new status
    status_display_names = {
        'completed': 'Completed',
        'in_progress': 'In Progress',
        'failed': 'Failed',
        'not_initiated': 'Not Initiated',
        'failed_to_fetch': 'Call Data Error' # Display name for failed to fetch
    }
    for status_key in ordered_statuses:
        if interview_status_counts[status_key] > 0:
            interview_status_labels.append(status_display_names[status_key])
            interview_status_data.append(interview_status_counts[status_key])

    interview_status_chart_data = {
        'labels': interview_status_labels,
        'data': interview_status_data,
    }

    # Chart 3: Hiring Recommendation Distribution
    hiring_recommendation_chart_data = {
        'labels': list(hiring_recommendation_counts.keys()),
        'data': list(hiring_recommendation_counts.values()),
    }

    # Chart 4: Candidates by Experience Level
    ordered_experience_labels = ['1 Year', '2 Years', '3 Years', '4 Years', '5-9 Years', '10+ Years']
    experience_chart_labels = []
    experience_chart_data_values = []
    for label in ordered_experience_labels:
        if experience_buckets[label] > 0:
            experience_chart_labels.append(label)
            experience_chart_data_values.append(experience_buckets[label])

    experience_chart_data = {
        'labels': experience_chart_labels,
        'data': experience_chart_data_values,
    }

    # 4. Prepare Context
    context = {
        # Key Metrics
        'total_candidates': total_candidates,
        'completed_calls': completed_calls,
        'in_progress_calls': in_progress_calls,
        'failed_calls': failed_calls,
        'success_rate': success_rate,

        # Chart Data
        'job_role_chart_data': job_role_chart_data,
        'interview_status_chart_data': interview_status_chart_data,
        'hiring_recommendation_chart_data': hiring_recommendation_chart_data,
        'experience_chart_data': experience_chart_data,
    }

    return render(request, 'dashboard.html', context)


@login_required
@user_passes_test(is_admin, login_url='/signin/') # Redirect to signin if not admin/superadmin
def admin_dashboard_view(request):
    """
    Admin-specific dashboard.
    """
    messages.info(request, f"Welcome, Admin {request.user.username}!")
    # You can add more admin-specific logic and data here
    return render(request, 'admin_dashboard.html')

@login_required
@user_passes_test(is_superadmin, login_url='/signin/') # Redirect to signin if not superadmin
def superadmin_dashboard_view(request):
    """
    Superadmin-specific dashboard.
    """
    messages.info(request, f"Welcome, Superadmin {request.user.username}!")
    # You can add more superadmin-specific logic and data here
    return render(request, 'superadmin_dashboard.html')


def all_job_descriptions(request):
    """
    Renders the page to display all uploaded and created job descriptions.
    """
    job_descriptions = JobDescriptionDocument.objects.all()
    context = {
        'job_descriptions': job_descriptions,
    }
    return render(request, 'all_job_descriptions.html', context)


def create_job_description(request):
    """
    Handles the creation of a new job description via text input with detailed IT fields.
    """
    if request.method == 'POST':
        title = request.POST.get('title')
        job_level = request.POST.get('job_level')
        department = request.POST.get('department')
        location = request.POST.get('location')
        employment_type = request.POST.get('employment_type')
        overview = request.POST.get('overview')
        responsibilities = request.POST.get('responsibilities')
        required_skills = request.POST.get('required_skills')
        preferred_skills = request.POST.get('preferred_skills')
        education_experience = request.POST.get('education_experience')
        benefits = request.POST.get('benefits')

        if title: # Only title is strictly required for text creation
            # Concatenate all text fields into a single string for file storage
            # This file will serve as a textual representation if no actual file is uploaded
            description_text_content = f"Title: {title}\n"
            if job_level: description_text_content += f"Job Level: {job_level.replace('_', ' ').title()}\n"
            if department: description_text_content += f"Department: {department}\n"
            if location: description_text_content += f"Location: {location}\n"
            if employment_type: description_text_content += f"Employment Type: {employment_type.replace('-', ' ').title()}\n"
            if overview: description_text_content += f"\nOverview:\n{overview}\n"
            if responsibilities: description_text_content += f"\nResponsibilities:\n{responsibilities}\n"
            if required_skills: description_text_content += f"\nRequired Skills:\n{required_skills}\n"
            if preferred_skills: description_text_content += f"\nPreferred Skills:\n{preferred_skills}\n"
            if education_experience: description_text_content += f"\nEducation & Experience:\n{education_experience}\n"
            if benefits: description_text_content += f"\nBenefits:\n{benefits}\n"

            # Create a unique filename for the text content
            file_name = f"{title.replace(' ', '_').lower()}_{JobDescriptionDocument.objects.count() + 1}_generated.txt"
            
            # Save the text content to a file in the default storage
            file_path = default_storage.save(f'job_descriptions/{file_name}', ContentFile(description_text_content.encode()))

            # Create a JobDescriptionDocument instance with all fields
            JobDescriptionDocument.objects.create(
                title=title,
                job_level=job_level,
                department=department,
                location=location,
                employment_type=employment_type,
                overview=overview,
                responsibilities=responsibilities,
                required_skills=required_skills,
                preferred_skills=preferred_skills,
                education_experience=education_experience,
                benefits=benefits,
                file=file_path # Assign the path to the file field
            )
            messages.success(request, 'Job description created successfully!')
            return redirect('all_job_descriptions')
        else:
            messages.error(request, 'Please provide at least a title for the job description.')

    return render(request, 'create_job_description.html')


def upload_job_description(request):
    """
    Handles the uploading of a new job description file.
    """
    if request.method == 'POST':
        title = request.POST.get('title')
        uploaded_file = request.FILES.get('file')

        if title and uploaded_file:
            # Check if the file is a PDF or DOCX (or other allowed types)
            if uploaded_file.content_type in ['application/pdf', 'application/msword', 'application/vnd.openxmlformats-officedocument.wordprocessingml.document', 'text/plain']:
                job_description = JobDescriptionDocument(title=title, file=uploaded_file)
                job_description.save()
                messages.success(request, 'Job description uploaded successfully!')
                return redirect('all_job_descriptions')
            else:
                messages.error(request, 'Invalid file type. Please upload a PDF, DOCX, or TXT file.')
        else:
            messages.error(request, 'Please provide both a title and a file.')

    return render(request, 'upload_job_description.html')

def edit_job_description(request, jd_id):
        """
        Handles editing an existing job description.
        You'll need to implement the form and saving logic here.
        """
        jd = get_object_or_404(JobDescriptionDocument, id=jd_id)
        if request.method == 'POST':
            # Handle form submission to update the JD
            jd.title = request.POST.get('title', jd.title)
            jd.job_level = request.POST.get('job_level', jd.job_level)
            jd.department = request.POST.get('department', jd.department)
            jd.location = request.POST.get('location', jd.location)
            jd.employment_type = request.POST.get('employment_type', jd.employment_type)
            jd.overview = request.POST.get('overview', jd.overview)
            jd.responsibilities = request.POST.get('responsibilities', jd.responsibilities)
            jd.required_skills = request.POST.get('required_skills', jd.required_skills)
            jd.preferred_skills = request.POST.get('preferred_skills', jd.preferred_skills)
            jd.education_experience = request.POST.get('education_experience', jd.education_experience)
            jd.benefits = request.POST.get('benefits', jd.benefits)
            # Handle file update if necessary (more complex, might involve deleting old and saving new)
            
            jd.save()
            messages.success(request, 'Job description updated successfully!')
            return redirect('analyze_jd', jd_id=jd.id) # Redirect to the detail page or all JDs

        context = {'jd': jd}
        return render(request, 'edit_job_description.html', context) # You'll create this template



@require_POST
def delete_jd(request, jd_id):
    """
    Deletes a job description document.
    """
    try:
        jd = get_object_or_404(JobDescriptionDocument, id=jd_id)
        # Delete the file from storage if it exists
        if jd.file:
            jd.file.delete(save=False) # save=False prevents an extra database save
        jd.delete()
        return JsonResponse({'success': True})
    except Exception as e:
        return JsonResponse({'success': False, 'error': str(e)})

def analyze_jd(request, jd_id):
    """
    Renders the page to display the full details of a job description.
    """
    jd = get_object_or_404(JobDescriptionDocument, id=jd_id)
    context = {'jd': jd}
    return render(request, 'analyze_jd.html', context)



def analyze_application_view(request, email_id, analysis_type, jd_id=None):
    """
    This function analyzes the resume content and returns a JSON response.
    This is where your logic for parsing and AI calls lives.
    """
    try:
        application = Application.objects.get(id=email_id)

        if not application.resume_url:
            return JsonResponse({'error': 'Resume file not found for this application.'}, status=404)

        resume_file_path = os.path.join(settings.BASE_DIR, application.resume_url.lstrip('/'))

        resume_content = ""
        if resume_file_path.endswith('.pdf'):
            try:
                doc = fitz.open(resume_file_path)
                resume_content = "".join(page.get_text() for page in doc)
                doc.close()
            except Exception as e:
                logger.error(f"Error reading PDF resume: {e}")
                return JsonResponse({'error': f"Error reading PDF resume: {e}"}, status=500)
        elif resume_file_path.endswith('.docx'):
            try:
                resume_content = docx2txt.process(resume_file_path)
            except Exception as e:
                logger.error(f"Error reading DOCX resume: {e}")
                return JsonResponse({'error': f"Error reading DOCX resume: {e}"}, status=500)

        job_description_content = ""
        job_title = "General"
        if jd_id:
            try:
                job_description_doc = JobDescriptionDocument.objects.get(id=jd_id)
                job_title = job_description_doc.job_title
                jd_file_path = os.path.join(settings.MEDIA_ROOT, 'job_descriptions', os.path.basename(job_description_doc.file.name))

                if jd_file_path.endswith('.pdf'):
                    try:
                        doc = fitz.open(jd_file_path)
                        job_description_content = "".join(page.get_text() for page in doc)
                        doc.close()
                    except Exception as e:
                        logger.error(f"Error reading PDF job description: {e}")
                elif jd_file_path.endswith('.docx'):
                    try:
                        job_description_content = docx2txt.process(jd_file_path)
                    except Exception as e:
                        logger.error(f"Error reading DOCX job description: {e}")
            except JobDescriptionDocument.DoesNotExist:
                logger.warning(f"Job Description with ID {jd_id} not found.")

        prompt = ""
        if analysis_type == 'basic':
            if job_description_content:
                prompt = f"""
                You are a resume screening bot. Please compare the following resume to the job description and provide a summary of the candidate's strengths and weaknesses for the role.
                Resume: {resume_content}
                Job Description: {job_description_content}
                """
            else:
                prompt = f"""
                You are an expert HR assistant. Analyze the following resume and provide a summary of the candidate's skills, experience, and key qualifications.
                Resume: {resume_content}
                """
        elif analysis_type == 'advanced':
            if not job_description_content:
                return JsonResponse({'error': 'Advanced analysis requires a job description.'}, status=400)
                
            prompt = f"""
            You are an advanced ATS (Applicant Tracking System) assistant. Perform a comprehensive analysis of the candidate's resume against the provided job description.
            
            Resume: {resume_content}
            
            Job Description: {job_description_content}
            
            Provide a detailed JSON object with the following keys:
            - 'match_score': An integer from 1 to 100 representing the relevance of the resume to the job description.
            - 'missing_keywords': A list of key skills or terms from the job description that were not found in the resume.
            - 'matching_keywords': A list of key skills or terms from the job description that were successfully found.
            - 'fit_summary': A detailed paragraph summarizing how well the candidate's experience and skills align with the job requirements.
            - 'areas_for_improvement': A list of suggestions for the candidate to improve their resume for this specific role.
            """

        # Perform the Gemini API call with the constructed prompt
        # Assuming `model` object from `genai.GenerativeModel` is available
        response = model.generate_content(prompt)
        
        if analysis_type == 'advanced':
            try:
                # Use a regular expression to find the JSON object and remove any surrounding text or formatting.
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    json_string = json_match.group(0)
                    analysis_data = json.loads(json_string)
                else:
                    raise json.JSONDecodeError("No JSON object found in response.", response.text, 0)
            except json.JSONDecodeError as e:
                # This is the improved error handling block
                logger.error(f"Failed to decode JSON from AI response: {response.text}. Error: {e}")
                return JsonResponse({
                    'error': 'Failed to parse AI response. The response was not valid JSON.',
                    'raw_ai_response': response.text
                }, status=500)
        else:
            analysis_data = {'summary': response.text}
            
        new_analysis = CandidateAnalysis.objects.create(
            full_name=application.candidate_name,
            job_role=job_title,
            resume_file_path=application.resume_url,
            analysis_summary=json.dumps(analysis_data)
        )

        return redirect('analysis_results', analysis_id=new_analysis.id)

    except Application.DoesNotExist:
        return JsonResponse({'error': 'Application not found.'}, status=404)
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        return JsonResponse({'error': f"An error occurred during analysis: {e}"}, status=500)
        
def analysis_results_view(request, analysis_id):
    """
    Displays the results of a stored analysis.
    """
    analysis = get_object_or_404(CandidateAnalysis, pk=analysis_id)
    
    try:
        # Check for and remove the markdown code block formatting
        summary_text = analysis.analysis_summary
        if summary_text.startswith('```json\n') and summary_text.endswith('\n```'):
            json_string = summary_text.strip('`\n') # Remove backticks and newlines
            json_string = json_string.lstrip('json').strip() # Remove the 'json' label
            result_data = json.loads(json_string)
        else:
            # If no markdown, assume it's already a clean JSON string
            result_data = json.loads(summary_text)
            
    except (json.JSONDecodeError, TypeError) as e:
        logger.error(f"Error decoding JSON for analysis_id {analysis_id}: {e}")
        return HttpResponseBadRequest("Error: Analysis data is corrupted.")

    context = {
        'analysis': analysis,
        'result_data': result_data
    }
    return render(request, 'analysis_results.html', context)


###################
def basic_resume_analysis_view(request):
    analysis_result = None
    resume_url = None
    
    # Get all existing job description documents
    job_description_documents = JobDescriptionDocument.objects.all()

    form = ResumeUploadForm(request.POST or None, request.FILES or None)

    if request.method == 'POST':
        logging.info("POST request received for resume analysis.")
        
        # Check if an existing JD was selected from the dropdown
        existing_jd_id = request.POST.get('job_description_id')
        
        if form.is_valid():
            logging.info("Form is valid. Processing uploaded files and form data.")
            resume_file = form.cleaned_data['resume_pdf']
            
            # Use uploaded job description file first, if it exists
            if 'job_description' in request.FILES:
                job_description_file = form.cleaned_data['job_description']
            elif existing_jd_id:
                # If an existing JD was selected, fetch it from the database
                try:
                    job_description_doc = JobDescriptionDocument.objects.get(pk=existing_jd_id)
                    job_description_file = job_description_doc.file
                    logging.info(f"Using existing job description from database: {job_description_doc.title}")
                except JobDescriptionDocument.DoesNotExist:
                    messages.error(request, "Selected job description not found.")
                    logging.error(f"Job Description with ID {existing_jd_id} not found.")
                    job_description_file = None
            else:
                messages.error(request, "Please upload or select a job description.")
                job_description_file = None
            
            job_role = form.cleaned_data['job_role']
            target_experience_type = form.cleaned_data['target_experience_type']
            min_years_required = form.cleaned_data['min_years_required']
            max_years_required = form.cleaned_data['max_years_required']

            if job_description_file:
                try:
                    # 1. Save the resume file for PDF preview
                    resume_filename = resume_storage.save(resume_file.name, resume_file)
                    resume_url = request.build_absolute_uri(resume_storage.url(resume_filename))
                    logging.info(f"Resume file '{resume_filename}' saved for preview. URL: {resume_url}")

                    # 2. Call the main AI analysis service function.
                    llm_response = services.analyze_resume_with_llm(
                        resume_file_obj=resume_file,
                        job_description_file_obj=job_description_file,
                        job_role=job_role,
                        experience_type=target_experience_type,
                        min_years=min_years_required,
                        max_years=max_years_required
                    )

                    if llm_response and not llm_response.get("error"):
                        analysis_result = llm_response
                        
                        # --- START: DATABASE SAVE LOGIC ---
                        try:
                            # Get the analysis_summary dictionary safely
                            analysis_summary = analysis_result.get("analysis_summary", {})
                            candidate_fitment_analysis = analysis_result.get("candidate_fitment_analysis", {})
                            
                            # Prepare data for the CandidateAnalysis model.
                            # Serialize complex data to JSON strings.
                            candidate_data_for_db = {
                                "resume_file_path": resume_filename,
                                "full_name": analysis_result.get("full_name"),
                                "job_role": job_role, # Use form data for this
                                "phone_no": analysis_result.get("contact_number"),
                                "hiring_recommendation": analysis_result.get("hiring_recommendation"),
                                "suggested_salary_range": analysis_result.get("suggested_salary_range"),
                                "interview_questions": json.dumps(analysis_result.get("interview_questions", [])),
                                
                                # Store the entire analysis_summary as a JSON string if needed,
                                # or remove this if all sub-components are stored separately
                                "analysis_summary": json.dumps(analysis_summary), 
                                
                                "experience_match": analysis_result.get("experience_match"),
                                "overall_experience": analysis_result.get("overall_experience"),
                                "current_company_name": analysis_result.get("current_company_name"),
                                "current_company_address": analysis_result.get("current_company_address"),
                                "fitment_verdict": analysis_result.get("fitment_verdict"), 
                                "aggregate_score": analysis_result.get("aggregate_score"), 
                                
                                # Fields extracted from candidate_fitment_analysis
                                "strategic_alignment": candidate_fitment_analysis.get("strategic_alignment", ""),
                                "quantifiable_impact": candidate_fitment_analysis.get("quantifiable_impact", ""),
                                "potential_gaps_risks": candidate_fitment_analysis.get("potential_gaps_risks", ""),
                                "comparable_experience": candidate_fitment_analysis.get("comparable_experience_analysis", ""), # Note the key name difference
                                
                                # Other top-level complex fields that need JSON dumping
                                "scoring_matrix_json": json.dumps(analysis_result.get("scoring_matrix", [])),
                                "bench_recommendation_json": json.dumps(analysis_result.get("bench_recommendation", {})),
                                "alternative_role_recommendations_json": json.dumps(analysis_result.get("alternative_role_recommendations", [])),
                                "automated_recruiter_insights_json": json.dumps(analysis_result.get("automated_recruiter_insights", {})),

                                # NEW: Fields extracted from analysis_summary
                                "candidate_overview": analysis_summary.get("candidate_overview", ""),
                                "technical_prowess_json": json.dumps(analysis_summary.get("technical_prowess", {})),
                                "project_impact_json": json.dumps(analysis_summary.get("project_impact", [])),
                                "education_certifications_json": json.dumps(analysis_summary.get("education_certifications", {})),
                                "overall_rating_summary": analysis_summary.get("overall_rating", ""), # Renamed to avoid conflict
                                "conclusion_summary": analysis_summary.get("conclusion", ""), # Renamed to avoid conflict
                            }
                            
                            # Use .create() to save the new object and get its automatically generated ID.
                            candidate_obj = CandidateAnalysis.objects.create(**candidate_data_for_db)
                            
                            # Now, update the analysis_result dictionary with the new ID
                            analysis_result['id'] = candidate_obj.id 
                            
                            messages.success(request, f"Analysis saved to database for {candidate_obj.full_name}.")
                        except Exception as db_save_error:
                            logging.warning(f"AI analysis completed, but failed to save to database: {db_save_error}")
                            messages.warning(request, f"AI analysis completed, but failed to save to database: {db_save_error}")
                            # Even if save fails, analysis_result is still available for the page, but without an id
                        # --- END: DATABASE SAVE LOGIC ---
                        
                        messages.success(request, f"AI analysis completed for {analysis_result.get('full_name', 'the candidate')}.")
                    else:
                        error_message = llm_response.get("error", "AI analysis failed to return a valid response.") if llm_response else "LLM response was empty or None."
                        logging.error(f"LLM response error: {error_message}")
                        messages.error(request, error_message)
                except Exception as e:
                    logging.error(f"An unexpected error occurred during the analysis process: {e}", exc_info=True)
                    messages.error(request, f"An unexpected error occurred during analysis: {e}")
            
        else:
            logging.warning("Form is not valid. Displaying errors.")
            messages.error(request, "Please correct the errors in the form before submitting.")

    context = {
        'form': form,
        'analysis_result': analysis_result,
        'resume_url': resume_url,
        'job_description_documents': job_description_documents, # Pass the documents to the template
    }

    return render(request, 'basic_resume_analysis.html', context)




def advance_resume_analysis_view(request):
    """
    This Django view handles the AI-based resume analysis process.
    It can be triggered automatically from a URL or manually via a form.

    Args:
        request: The Django HttpRequest object.

    Returns:
        A Django HttpResponse that renders the analysis results page.
    """
    analysis_result = None
    resume_url = None
    job_description_documents = JobDescriptionDocument.objects.all()

    # Get application ID and job description ID from URL query parameters
    application_id = request.GET.get('application_id')
    job_description_id = request.GET.get('job_description_id')
    
    application = None
    if application_id:
        try:
            application = Application.objects.get(id=application_id)
            if application.resume_url:
                resume_url = application.resume_url
        except Application.DoesNotExist:
            messages.error(request, "Application not found.")
            logging.error(f"Application with ID {application_id} not found.")

    # Check if analysis should be performed automatically
    if application_id and job_description_id:
        logging.info("Automatic analysis triggered from application list.")
        
        resume_file = None
        job_description_file = None

        try:
            # Construct file path from application's resume URL
            relative_path = application.resume_url.lstrip('/').replace('media/', '', 1)
            file_path = os.path.join(settings.MEDIA_ROOT, relative_path)
            
            # This part simulates opening the file from the filesystem.
            # In a production environment, you might use Django's storage API.
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    resume_file = SimpleUploadedFile(
                        name=os.path.basename(file_path),
                        content=f.read(),
                        content_type='application/pdf'
                    )
                logging.info(f"Using resume file from application ID: {application.id}.")
            else:
                messages.error(request, "Resume file not found for the selected application.")
                logging.error(f"Resume file not found at path: {file_path}")
            
            # Get the job description file object from the database
            job_description_doc = JobDescriptionDocument.objects.get(pk=job_description_id)
            job_description_file = job_description_doc.file
        
            if resume_file and job_description_file:
                job_role = application.subject
                min_years_required = application.experience if application.experience is not None else 0
                
                # Assign default values for the missing arguments as in your code
                experience_type = 'range'
                max_years = min_years_required + 5  # Set a reasonable upper range
                
                # Call the LLM service
                llm_response = services.analyze_resume_with_llm(
                    resume_file_obj=resume_file,
                    job_description_file_obj=job_description_file,
                    job_role=job_role,
                    experience_type=experience_type,
                    min_years=min_years_required,
                    max_years=max_years,
                )
                
                if llm_response and not llm_response.get("error"):
                    analysis_result = llm_response
                    messages.success(request, f"AI analysis completed for {analysis_result.get('full_name', 'the candidate')}.")
                else:
                    error_message = llm_response.get("error", "AI analysis failed.")
                    messages.error(request, error_message)

        except JobDescriptionDocument.DoesNotExist:
            messages.error(request, "Selected job description not found.")
            logging.error(f"Job description with ID {job_description_id} not found.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during analysis: {e}", exc_info=True)
            messages.error(request, f"An unexpected error occurred: {e}")
            
    # Handle GET or POST for initial form display or manual submission
    form_initial_data = {}
    if application:
        form_initial_data = {
            'job_role': application.subject if application.subject else '',
            'min_years_required': application.experience if application.experience is not None else 0,
            'application_id': application.id,
        }
        
    # The form is no longer needed for automatic analysis but is kept for context
    form = ResumeUploadForm(request.POST or None, request.FILES or None, initial=form_initial_data)

    if request.method == 'POST' and form.is_valid():
        pass # This section is now vestigial but can be left in for future use.

    context = {
        'form': form,
        'analysis_result': analysis_result,
        'resume_url': resume_url,
        'job_description_documents': job_description_documents,
    }

    return render(request, 'advance_resume_analysis.html', context)
