# # # myhrproject/hr_app/models.py

# from django.db import models
# import json # Import json to handle JSONField serialization/deserialization
# from django.contrib.auth.models import User

# # class CandidateAnalysis(models.Model):
# #     """
# #     Model to store the results of AI resume analysis for a candidate.
# #     """
# #     full_name = models.CharField(max_length=255, blank=True, null=True)
# #     job_role = models.CharField(max_length=255, blank=True, null=True)
# #     phone_no = models.CharField(max_length=20, blank=True, null=True)
# #     overall_experience = models.CharField(max_length=50, blank=True, null=True) # e.g., "5 years", "6 months"
# #     hiring_recommendation = models.CharField(max_length=20, blank=True, null=True) # "Hire", "Resign", "Reject"
# #     experience_match = models.CharField(max_length=50, blank=True, null=True) # "Good Match", "Underqualified", "Overqualified"
# #     suggested_salary_range = models.CharField(max_length=50, blank=True, null=True)
# #     final_decision = models.CharField(max_length=20, blank=True, null=True) # "Selected", "Not Selected"
# #     final_salary = models.DecimalField(max_digits=10, decimal_places=2, blank=True, null=True)
# #     analysis_summary = models.TextField(blank=True, null=True) # Detailed AI evaluation
    
# #     # JSONField is ideal for storing lists/dictionaries directly in the database (e.g., interview questions)
# #     interview_questions = models.JSONField(blank=True, null=True) 
    
# #     timestamp = models.DateTimeField(auto_now_add=True) # Automatically sets the creation timestamp
    
# #     # Fields for current company information
# #     current_company_name = models.CharField(max_length=255, blank=True, null=True)
# #     current_company_address = models.CharField(max_length=255, blank=True, null=True)

# #     # Field to store the Bland.ai call ID for tracking interviews
# #     bland_call_id = models.CharField(max_length=255, unique=True, blank=True, null=True) 

# #     def __str__(self):
# #         """String representation of the CandidateAnalysis object."""
# #         return f"{self.full_name} ({self.job_role})"

# #     class Meta:
# #         """Meta options for the model."""
# #         verbose_name = "Candidate Analysis"
# #         verbose_name_plural = "Candidate Analyses"
# #         # Ordering by timestamp in descending order by default
# #         ordering = ['-timestamp']
# class User(AbstractUser):
#     """
#     Custom User model extending AbstractUser to add a 'role' field.
#     Roles can be 'user', 'admin', or 'superadmin'.
#     """
#     ROLE_CHOICES = (
#         ('user', 'User'),
#         ('admin', 'Admin'),
#         ('superadmin', 'Superadmin'),
#     )
#     role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')

#     def __str__(self):
#         return self.username


# class CandidateAnalysis(models.Model):
#     # ... your existing fields ...
#     full_name = models.CharField(max_length=255)
#     job_role = models.CharField(max_length=255)
#     # ... other fields ...
#     phone_no = models.CharField(max_length=20, blank=True, null=True)
#     bland_call_id = models.CharField(max_length=100, blank=True, null=True)
#     hiring_recommendation = models.CharField(max_length=50, blank=True, null=True)
#     suggested_salary_range = models.CharField(max_length=100, blank=True, null=True)
#     interview_questions = models.TextField(blank=True, null=True) # If you store questions
#     analysis_summary = models.TextField(blank=True, null=True)
#     experience_match = models.TextField(blank=True, null=True)
#     overall_experience = models.CharField(max_length=50, blank=True, null=True)
#     current_company_name = models.CharField(max_length=255, blank=True, null=True)
#     current_company_address = models.CharField(max_length=255, blank=True, null=True)
#     final_decision = models.CharField(max_length=50, blank=True, null=True)
#     final_salary = models.CharField(max_length=100, blank=True, null=True)

#     timestamp = models.DateTimeField(auto_now_add=True) # Creation timestamp
#     last_updated = models.DateTimeField(auto_now=True)   # Automatically updates on each save

#     def __str__(self):
#         return self.full_name

    
#     def objects(self):
#         pass
    
    
    

    
# class When:
    
#     def __init__(self, hiring_recommendation, then):
#         pass




    # myhrproject/hr_app/models.py

  # myhrproject/hr_app/models.py

from django.db import models
import json # Import json to handle JSONField serialization/deserialization
from django.contrib.auth.models import AbstractUser # Import AbstractUser

# Extend Django's built-in User model
class User(AbstractUser):
    """
    Custom User model extending AbstractUser to add a 'role' field.
    Roles can be 'user', 'admin', or 'superadmin'.
    """
    ROLE_CHOICES = (
        ('user', 'User'),
        ('admin', 'Admin'),
        ('superadmin', 'Superadmin'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')

    # Add unique related_name arguments to avoid clashes with auth.User
    groups = models.ManyToManyField(
        'auth.Group',
        verbose_name='groups',
        blank=True,
        help_text='The groups this user belongs to. A user will get all permissions granted to each of their groups.',
        related_name="hr_app_user_set", # Unique related_name for groups
        related_query_name="hr_app_user",
    )
    # User permissions field re-added with related_name to resolve E304 error.
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        verbose_name='user permissions',
        blank=True,
        help_text='Specific permissions for this user.',
        related_name="hr_app_user_permissions_set", # Unique related_name for user_permissions
        related_query_name="hr_app_user",
    )

    def __str__(self):
        return self.username

class CandidateAnalysis(models.Model):
    # Existing fields (ensure these are present and match your current model)
    full_name = models.CharField(max_length=255, null=True, blank=True)
    job_role = models.CharField(max_length=255, null=True, blank=True)
    phone_no = models.CharField(max_length=50, null=True, blank=True)
    hiring_recommendation = models.CharField(max_length=50, null=True, blank=True)
    suggested_salary_range = models.CharField(max_length=100, null=True, blank=True)
    interview_questions = models.TextField(null=True, blank=True) # Stores JSON string

    # This field was previously storing the entire summary, now it's optional
    # You can keep it if you want a redundant full summary, or remove it if all sub-components are separate
    analysis_summary = models.TextField(null=True, blank=True) # Stores JSON string of the full summary

    experience_match = models.CharField(max_length=50, null=True, blank=True)
    overall_experience = models.CharField(max_length=50, null=True, blank=True)
    current_company_name = models.CharField(max_length=255, null=True, blank=True)
    current_company_address = models.CharField(max_length=255, null=True, blank=True)

    # Fields that were causing the "unexpected keyword argument" error
    fitment_verdict = models.CharField(max_length=50, null=True, blank=True)
    aggregate_score = models.CharField(max_length=50, null=True, blank=True)

    # NEW FIELDS FOR FINAL DECISION AND SALARY
    final_decision = models.CharField(max_length=50, null=True, blank=True)
    final_salary = models.IntegerField(null=True, blank=True) # Using IntegerField for salary

    # Fields extracted from candidate_fitment_analysis
    strategic_alignment = models.TextField(null=True, blank=True)
    quantifiable_impact = models.TextField(null=True, blank=True)
    potential_gaps_risks = models.TextField(null=True, blank=True)
    comparable_experience = models.TextField(null=True, blank=True) # Note: this was `comparable_experience_analysis` in LLM response

    # Other top-level complex fields that need JSON dumping
    scoring_matrix_json = models.TextField(null=True, blank=True) # Stores JSON string
    bench_recommendation_json = models.TextField(null=True, blank=True) # Stores JSON string
    alternative_role_recommendations_json = models.TextField(null=True, blank=True) # Stores JSON string
    automated_recruiter_insights_json = models.TextField(null=True, blank=True) # Stores JSON string

    # Fields extracted from analysis_summary (NEWLY ADDED IN VIEWS.PY)
    candidate_overview = models.TextField(null=True, blank=True)
    technical_prowess_json = models.TextField(null=True, blank=True) # Stores JSON string
    project_impact_json = models.TextField(null=True, blank=True) # Stores JSON string
    education_certifications_json = models.TextField(null=True, blank=True) # Stores JSON string
    overall_rating_summary = models.CharField(max_length=50, null=True, blank=True) # Renamed to avoid conflict
    conclusion_summary = models.TextField(null=True, blank=True) # Renamed to avoid conflict
    bland_call_id = models.CharField(max_length=100, blank=True, null=True)
    interview_status = models.CharField(max_length=50, null=True, blank=True, default='Pending')
    resume_file_path = models.CharField(max_length=255, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    DECISION_CHOICES = [
        ('shortlisted', 'Shortlisted'),
        ('selected', 'Selected'),
        ('not_selected', 'Not Selected'),
    ]
    
    final_decision = models.CharField(
        max_length=20,
        choices=DECISION_CHOICES,
        default='shortlisted',
        blank=True,
        null=True
    )
    
    final_salary = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        blank=True,
        null=True
    )

    def __str__(self):
        return self.full_name


class When:
    
    def __init__(self, hiring_recommendation, then):
        pass

from django.db import models

class Application(models.Model):
    """
    Model to store extracted information from incoming job applications (resumes).
    """
    candidate_name = models.CharField(max_length=255, blank=True, null=True)
    from_email = models.EmailField(max_length=255, blank=True, null=True, help_text="Email address of the sender.")
    delivery_date = models.DateTimeField(blank=True, null=True, help_text="Date and time the application was received.")
    experience = models.IntegerField(blank=True, null=True, help_text="Total years of professional experience.")
    mobile_number = models.CharField(max_length=50, blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    email_address = models.EmailField(max_length=255, blank=True, null=True, help_text="Email address found in the resume.")
    subject = models.CharField(max_length=500, blank=True, null=True)
    resume_url = models.URLField(max_length=1000, blank=True, null=True, unique=True, help_text="URL to the downloaded resume file.")

    # Timestamps for tracking when the record was created/updated
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        # Default ordering: newest applications first
        ordering = ['-delivery_date', '-created_at']
        verbose_name = "Incoming Application"
        verbose_name_plural = "Incoming Applications"

    def __str__(self):
        return f"{self.candidate_name or 'Unknown Candidate'} - {self.subject or 'No Subject'}"


from django.db import models
from django.core.files.storage import FileSystemStorage
job_description_storage = FileSystemStorage(location='media/job_descriptions')
resume_storage = FileSystemStorage(location='media/resumes')


class JobDescriptionDocument(models.Model):
    """
    Model to store uploaded job description documents for reuse,
    and job descriptions created directly via text input, with IT-specific fields.
    """
    title = models.CharField(max_length=255, help_text="A descriptive title for the job description.")
    
    # New IT-specific fields
    job_level_choices = [
        ('intern', 'Intern'),
        ('junior', 'Junior'),
        ('mid', 'Mid-Level'),
        ('senior', 'Senior'),
        ('lead', 'Lead'),
        ('manager', 'Manager'),
        ('director', 'Director'),
        ('vp', 'Vice President'),
    ]
    job_level = models.CharField(
        max_length=50,
        choices=job_level_choices,
        default='mid',
        help_text="The experience level required for this position.",
        null=True, blank=True
    )
    department = models.CharField(max_length=100, help_text="The department or team this role belongs to.", null=True, blank=True)
    location = models.CharField(max_length=100, help_text="The primary work location (e.g., City, State, Remote).", null=True, blank=True)
    employment_type_choices = [
        ('full-time', 'Full-time'),
        ('part-time', 'Part-time'),
        ('contract', 'Contract'),
        ('temporary', 'Temporary'),
        ('internship', 'Internship'),
    ]
    employment_type = models.CharField(
        max_length=50,
        choices=employment_type_choices,
        default='full-time',
        help_text="Type of employment (e.g., Full-time, Contract).",
        null=True, blank=True
    )
    
    overview = models.TextField(help_text="A brief overview of the role and its purpose.", null=True, blank=True)
    responsibilities = models.TextField(help_text="Key duties and responsibilities for this role.", null=True, blank=True)
    required_skills = models.TextField(help_text="Mandatory skills and qualifications (e.g., Python, AWS, Agile).", null=True, blank=True)
    preferred_skills = models.TextField(help_text="Desirable but not mandatory skills.", null=True, blank=True)
    education_experience = models.TextField(help_text="Required education and work experience.", null=True, blank=True)
    benefits = models.TextField(help_text="Company benefits and perks.", null=True, blank=True)
    
    # Original file field, now optional
    file = models.FileField(storage=job_description_storage, 
                            help_text="The uploaded job description file (optional if created via text).",
                            null=True, blank=True)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = "Job Description Document"
        verbose_name_plural = "Job Description Documents"
        ordering = ['-uploaded_at']
