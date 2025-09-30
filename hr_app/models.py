from django.conf import settings
from django.db import models
import json # Import json to handle JSONField serialization/deserialization
from django.contrib.auth.models import AbstractUser # Import AbstractUser
from django.core.mail import get_connection
import os
# Extend Django's built-in User model
class User(AbstractUser):
    """
    Custom User model extending AbstractUser to add a 'role' field and a 'created_by' field.
    Roles can be 'user', 'admin', or 'superadmin'.
    """
    ROLE_CHOICES = (
        ('user', 'User'),
        ('admin', 'Admin'),
        ('superadmin', 'Superadmin'),
    )
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='user')

    # --- Add this new field to your model ---
    # This ForeignKey links a user to the admin who created them.
    # We use 'self' because the link is to the same User model.
    created_by = models.ForeignKey(
        'self',
        on_delete=models.SET_NULL,  # If the creator is deleted, this field becomes NULL
        null=True,                  # The field can be empty (for superadmins and users not created by an admin)
        blank=True,                 # The field is not required in forms
        related_name='created_users' # A custom name to avoid clashes and for easy lookups
    )
    # --- End of new field ---

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

# class CandidateAnalysis(models.Model):
#     # Existing fields (ensure these are present and match your current model)
#     user = models.ForeignKey(User, on_delete=models.CASCADE)
#     full_name = models.CharField(max_length=255, null=True, blank=True)
#     job_role = models.CharField(max_length=255, null=True, blank=True)
#     phone_no = models.CharField(max_length=50, null=True, blank=True)
#     hiring_recommendation = models.CharField(max_length=50, null=True, blank=True)
#     suggested_salary_range = models.CharField(max_length=100, null=True, blank=True)
#     interview_questions = models.TextField(null=True, blank=True) # Stores JSON string

#     # This field was previously storing the entire summary, now it's optional
#     # You can keep it if you want a redundant full summary, or remove it if all sub-components are separate
#     analysis_summary = models.TextField(null=True, blank=True) # Stores JSON string of the full summary

#     experience_match = models.CharField(max_length=50, null=True, blank=True)
#     overall_experience = models.CharField(max_length=50, null=True, blank=True)
#     current_company_name = models.CharField(max_length=255, null=True, blank=True)
#     current_company_address = models.CharField(max_length=255, null=True, blank=True)

#     # Fields that were causing the "unexpected keyword argument" error
#     fitment_verdict = models.CharField(max_length=50, null=True, blank=True)
#     aggregate_score = models.CharField(max_length=50, null=True, blank=True)

#     # NEW FIELDS FOR FINAL DECISION AND SALARY
#     final_decision = models.CharField(max_length=50, null=True, blank=True)
#     final_salary = models.IntegerField(null=True, blank=True) # Using IntegerField for salary

#     # Fields extracted from candidate_fitment_analysis
#     strategic_alignment = models.TextField(null=True, blank=True)
#     quantifiable_impact = models.TextField(null=True, blank=True)
#     potential_gaps_risks = models.TextField(null=True, blank=True)
#     comparable_experience = models.TextField(null=True, blank=True) # Note: this was `comparable_experience_analysis` in LLM response

#     # Other top-level complex fields that need JSON dumping
#     scoring_matrix_json = models.TextField(null=True, blank=True) # Stores JSON string
#     bench_recommendation_json = models.TextField(null=True, blank=True) # Stores JSON string
#     alternative_role_recommendations_json = models.TextField(null=True, blank=True) # Stores JSON string
#     automated_recruiter_insights_json = models.TextField(null=True, blank=True) # Stores JSON string
    
#     ai_summary = models.TextField(blank=True, null=True)
#     confidence_score = models.IntegerField(blank=True, null=True)
#     suggested_questions = models.TextField(blank=True, null=True)
#     # Fields extracted from analysis_summary (NEWLY ADDED IN VIEWS.PY)
#     candidate_overview = models.TextField(null=True, blank=True)
#     technical_prowess_json = models.TextField(null=True, blank=True) # Stores JSON string
#     project_impact_json = models.TextField(null=True, blank=True) # Stores JSON string
#     education_certifications_json = models.TextField(null=True, blank=True) # Stores JSON string
#     overall_rating_summary = models.CharField(max_length=50, null=True, blank=True) # Renamed to avoid conflict
#     conclusion_summary = models.TextField(null=True, blank=True) # Renamed to avoid conflict
#     bland_call_id = models.CharField(max_length=100, blank=True, null=True)
#     interview_status = models.CharField(max_length=50, null=True, blank=True, default='Pending')
#     resume_file_path = models.CharField(max_length=255, null=True, blank=True)

#     created_at = models.DateTimeField(auto_now_add=True)
#     updated_at = models.DateTimeField(auto_now=True)
    
#     DECISION_CHOICES = [
#         ('shortlisted', 'Shortlisted'),
#         ('selected', 'Selected'),
#         ('not_selected', 'Not Selected'),
#     ]
    
#     final_decision = models.CharField(
#         max_length=20,
#         choices=DECISION_CHOICES,
#         default='Pending',
#         blank=True,
#         null=True
#     )
    
#     final_salary = models.DecimalField(
#         max_digits=10, 
#         decimal_places=2,
#         blank=True,
#         null=True
#     )
#     ANALYSIS_TYPES = (
#         ('Manual', 'Manual ATS Analysis'),
#         ('Basic', 'Basic ATS Analysis'),
#         ('Advance', 'Advanced ATS Analysis'),
#     )
#     analysis_type = models.CharField(
#         max_length=10, 
#         choices=ANALYSIS_TYPES, 
#         default='Manual', 
#         help_text="Type of resume analysis (Basic or Advanced)."
#     )



#     def __str__(self):
#         return self.full_name


# class When:
    
#     def __init__(self, hiring_recommendation, then):
#         pass

from django.db import models

class Application(models.Model):
    """
    Model to store extracted information from incoming job applications (resumes).
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    candidate_name = models.CharField(max_length=255, blank=True, null=True)
    from_email = models.EmailField(max_length=255, blank=True, null=True, help_text="Email address of the sender.")
    delivery_date = models.DateTimeField(blank=True, null=True, help_text="Date and time the application was received.")
    experience = models.IntegerField(blank=True, null=True, help_text="Total years of professional experience.")
    mobile_number = models.CharField(max_length=50, blank=True, null=True)
    location = models.CharField(max_length=255, blank=True, null=True)
    email_address = models.EmailField(max_length=255, blank=True, null=True, help_text="Email address found in the resume.")
    subject = models.CharField(max_length=500, blank=True, null=True)
    resume_url = models.URLField(max_length=1000, blank=True, null=True, unique=True, help_text="URL to the downloaded resume file.")
    remark = models.CharField(max_length=500, blank=True, null=True)
    job_role = models.CharField(max_length=255, blank=True, null=True)

    # --- New fields for ATS analysis ---
    analysis_type = models.CharField(max_length=20, blank=True, null=True, help_text="Type of analysis performed: basic or advanced.")
    job_description = models.ForeignKey('JobDescriptionDocument', on_delete=models.SET_NULL, null=True, blank=True, help_text="The job description associated with this application.")
    match_score = models.DecimalField(max_digits=5, decimal_places=2, blank=True, null=True, help_text="Match score from the advanced analysis.")
    match_summary = models.TextField(blank=True, null=True, help_text="Summary of the match between the resume and job description.")
    
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
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    title = models.CharField(max_length=255, help_text="A descriptive title for the job description.")
    company_name = models.CharField(max_length=255, help_text="The name of the company posting the job.", null=True, blank=True)
    
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
    
    # Location fields
    country = models.CharField(max_length=100, help_text="The country for the job location.", null=True, blank=True)
    state = models.CharField(max_length=100, help_text="The state for the job location.", null=True, blank=True)
    city = models.CharField(max_length=100, help_text="The city for the job location.", null=True, blank=True)

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

    # Salary fields
    salary_min = models.IntegerField(null=True, blank=True, help_text="The minimum salary for the position.")
    salary_max = models.IntegerField(null=True, blank=True, help_text="The maximum salary for the position.")
    salary_frequency_choices = [
        ('hourly', 'Hourly'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('yearly', 'Yearly'),
    ]
    salary_frequency = models.CharField(
        max_length=50,
        choices=salary_frequency_choices,
        help_text="The frequency of the salary payment (e.g., Yearly, Hourly).",
        null=True, blank=True
    )
    
    overview = models.TextField(help_text="A brief overview of the role and its purpose.", null=True, blank=True)
    responsibilities = models.TextField(help_text="Key duties and responsibilities for this role.", null=True, blank=True)
    required_skills = models.TextField(help_text="Mandatory skills and qualifications (e.g., Python, AWS, Agile).", null=True, blank=True)
    preferred_skills = models.TextField(help_text="Desirable but not mandatory skills.", null=True, blank=True)
    education_experience = models.TextField(help_text="Required education and work experience.", null=True, blank=True)
    benefits = models.TextField(help_text="Company benefits and perks.", null=True, blank=True)
    
    # Renamed from 'description' to 'job_description' for clarity and form compatibility
    job_description = models.TextField(help_text="The full, detailed description of the job role.", null=True, blank=True)

    # Original file field, now optional
    file = models.FileField(
        upload_to="job_descriptions/",
        storage=job_description_storage,
        help_text="The uploaded job description file (optional if created via text).",
        null=True, blank=True
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.title

    class Meta:
        verbose_name = "Job Description Document"
        verbose_name_plural = "Job Description Documents"
        ordering = ['-uploaded_at']


class Job(models.Model):
    """
    Represents a job description. This is assumed to be the 'JobDescriptionDocument'
    that the JobPosting model refers to.
    """
    JOB_LEVEL_CHOICES = [
        ('entry', 'Entry Level'),
        ('mid', 'Mid-Level'),
        ('senior', 'Senior Level'),
        ('executive', 'Executive'),
    ]

    title = models.CharField(max_length=200)
    job_level = models.CharField(max_length=10, choices=JOB_LEVEL_CHOICES, default='mid')
    company_name = models.CharField(max_length=200)
    city = models.CharField(max_length=100)
    state = models.CharField(max_length=100)
    country = models.CharField(max_length=100)
    description = models.TextField()

    def __str__(self):
        return f"{self.title} at {self.company_name}"

class JobPosting(models.Model):
    """
    Model to track individual job postings on different platforms.
    """
    job_description = models.ForeignKey(
        Job, 
        on_delete=models.CASCADE, 
        related_name='postings',
        help_text="The job description this posting is for."
    )
    platform_choices = [
        ('linkedin', 'LinkedIn'),
        ('indeed', 'Indeed'),
        ('glassdoor', 'Glassdoor'),
        ('shine', 'Shine'),
        ('naukri', 'Naukri'),
        ('company_website', 'Company Website'),
        ('other', 'Other'),
    ]
    platform = models.CharField(
        max_length=50,
        choices=platform_choices,
        help_text="The platform where the job was posted."
    )
    posting_url = models.URLField(
        max_length=500,
        null=True, blank=True,
        help_text="The direct URL to the job posting on the platform."
    )
    posted_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.job_description.title} on {self.get_platform_display()}"
    
    def get_platform_display(self):
        """
        Returns the human-readable platform name.
        """
        return dict(self.platform_choices)[self.platform]


class EmailConfiguration(models.Model):
    """
    A model to store dynamic email configuration settings,
    linked to a specific user.
    """
    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, primary_key=True)
    
    # SMTP Configuration (for sending emails)
    email_host = models.CharField(max_length=255, help_text="SMTP server address (e.g., smtp.gmail.com)")
    email_port = models.IntegerField(help_text="SMTP port number (e.g., 587)", null=True, blank=True, default=587)
    
    # IMAP Configuration (for receiving emails)
    imap_host = models.CharField(max_length=255, help_text="IMAP server address (e.g., imap.gmail.com)", default="imap.zoho.in")
    imap_port = models.IntegerField(help_text="IMAP port number (e.g., 993 for SSL)", default=993)
    
    # Authentication details for both protocols
    email_host_user = models.CharField(max_length=255, help_text="Email address for authentication")
    email_host_password = models.CharField(max_length=255, help_text="Password or app-specific password for the email account")
    email_use_tls = models.BooleanField(default=True, help_text="Use a TLS secure connection")
    email_use_ssl = models.BooleanField(default=False, help_text="Use a SSL secure connection")
    email_from = models.CharField(max_length=255, blank=True, help_text="Sender's email address. If blank, uses the host user.")

    class Meta:
        verbose_name = "Email Configuration"

    def __str__(self):
        return f"Email Configuration for {self.user.username}"

    def get_connection(self):
        """
        Returns a Django email connection object using the settings from this model instance.
        """
        return get_connection(
            host=self.email_host,
            port=self.email_port,
            username=self.email_host_user,
            password=self.email_host_password,
            use_tls=self.email_use_tls,
            use_ssl=self.email_use_ssl,
        )


class SentEmail(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    # user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='sent_emails')
    recipient_emails = models.TextField()
    subject = models.CharField(max_length=255)
    body = models.TextField()
    sent_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Email to {self.recipient_emails} sent by {self.user.username}"
    
class DraftEmail(models.Model):
    """
    Saves and manages email drafts for a user.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    recipient_emails = models.TextField(blank=True)
    subject = models.CharField(max_length=255, blank=True)
    body = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    last_modified = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Draft: {self.subject[:50]}..." if self.subject else "Untitled Draft"



# class CareerPage(models.Model):
#     """
#     Represents a single career page in the portal.
#     """
#     title = models.CharField(max_length=200)
#     description = models.TextField()
#     is_active = models.BooleanField(default=True)
#     date_posted = models.DateTimeField(auto_now_add=True)

#     def __str__(self):
#         return self.title

#     class Meta:
#         ordering = ['-date_posted']
from django.core.exceptions import ValidationError
class ThemeSettings(models.Model):
    """Stores theme colors for each user (One-to-One with User)."""

    user = models.OneToOneField(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="theme_settings"
    )

    # --- Background Colors ---
    theme_primary_color = models.CharField(
        max_length=7,
        default="#3e4f47",
        help_text="Primary color for buttons and highlights."
    )
    theme_secondary_color = models.CharField(
        max_length=7,
        default="#eef2f6",
        help_text="Secondary/light background color (e.g., footer)."
    )
    theme_background_color = models.CharField(
        max_length=7,
        default="#f7f9fc",
        help_text="Main page background color."
    )

    # --- Text Colors for the corresponding backgrounds ---
    theme_primary_color_text = models.CharField(
        max_length=7,
        default="#ffffff",
        help_text="Text color on Primary background (e.g., button text)."
    )
    theme_secondary_color_text = models.CharField(
        max_length=7,
        default="#333333",
        help_text="Text color on Secondary background (e.g., light sections/footer links)."
    )
    theme_background_color_text = models.CharField(
        max_length=7,
        default="#333333",
        help_text="Default text color on the main Background."
    )

    def __str__(self):
        return f"ThemeSettings for {self.user.username}"



class CareerPage(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='job_postings', null=True, blank=True)
    category = models.ForeignKey('Category', on_delete=models.SET_NULL, null=True, blank=True, related_name='jobs')
    title = models.CharField(max_length=200)
    company = models.CharField(max_length=100, default='Our Company')
    company_logo = models.ImageField(upload_to='company_logos/', blank=True, null=True)
    date_posted = models.DateTimeField(auto_now_add=True)
    location = models.CharField(max_length=100)
    job_type = models.CharField(max_length=50)
    category = models.CharField(max_length=50)
    experience = models.CharField(max_length=50, blank=True, null=True)
    salary = models.CharField(max_length=100)
    is_active = models.BooleanField(default=True)
    description = models.TextField()
    about_company = models.TextField(blank=True, null=True)
    skills = models.TextField(blank=True, help_text='Comma-separated skills')
    benefits = models.TextField(blank=True, null=True)
    application_link = models.URLField(max_length=500, default='#')
    responsibilities = models.TextField(blank=True, null=True)
    qualifications = models.TextField(blank=True, null=True)
    date_line = models.DateField(blank=True, null=True)

    class Meta:
        ordering = ['-date_posted']

    def __str__(self):
        return self.title

class CompanyInfo(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, unique=True, related_name='company_info')
    company_name = models.CharField(max_length=200, default='Your Company Name')
    slider_header1 = models.CharField(max_length=1000, default='Find The Perfect Job That You Deserved')
    slider_header2 = models.CharField(max_length=1000, default='Find The Best Startup Job That Fit You')
    slider_paragraph1 = models.CharField(max_length=1000, default='Vero elitr justo clita lorem. Ipsum dolor at sed stet sit diam no. Kasd rebum ipsum et diam justo clita et kasd rebum sea elitr.')
    slider_paragraph2 = models.CharField(max_length=1000, default='Vero elitr justo clita lorem. Ipsum dolor at sed stet sit diam no. Kasd rebum ipsum et diam justo clita et kasd rebum sea elitr.')
    address = models.CharField(max_length=255, blank=True, null=True)
    phone_number = models.CharField(max_length=20, blank=True, null=True)
    email = models.EmailField(blank=True, null=True)
    about_us_url = models.URLField(max_length=200, blank=True, null=True)
    contact_us_url = models.URLField(max_length=200, blank=True, null=True)
    our_services_url = models.URLField(max_length=200, blank=True, null=True)
    privacy_policy_url = models.URLField(max_length=200, blank=True, null=True)
    terms_and_conditions_url = models.URLField(max_length=200, blank=True, null=True)
    company_logo = models.ImageField(upload_to='company_logos/', blank=True, null=True)
    application_url = models.URLField(max_length=500, default='#')

    # Social media URLs
    twitter_url = models.URLField(max_length=200, blank=True, null=True)
    facebook_url = models.URLField(max_length=200, blank=True, null=True)
    youtube_url = models.URLField(max_length=200, blank=True, null=True)
    linkedin_url = models.URLField(max_length=200, blank=True, null=True)

    # NEW FIELD → Enable/Disable Career Page
    career_page_enabled = models.BooleanField(default=True)

    class Meta:
        verbose_name_plural = "Company Information"

    def __str__(self):
        return self.company_name

# This is your existing Apply_career model, with the new fields added
class Apply_career(models.Model):
    """
    Model for job applications submitted by users.
    """
    # This foreign key links an application to a specific job posting
    career = models.ForeignKey(CareerPage, on_delete=models.CASCADE)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='applications', blank=True, null=True)
    
    first_name = models.CharField(max_length=100)
    last_name = models.CharField(max_length=100)
    email = models.EmailField()
    phone = models.CharField(max_length=20, blank=True, null=True)
    
    # New Fields
    experience = models.IntegerField(verbose_name='Total Experience in Years', blank=True, null=True)
    current_ctc = models.CharField(max_length=50, verbose_name='Current CTC (Annual)', blank=True, null=True)
    expected_ctc = models.CharField(max_length=50, verbose_name='Expected CTC (Annual)', blank=True, null=True)
    qualification = models.CharField(max_length=255, verbose_name='Highest Qualification', blank=True, null=True)
    notice_period = models.CharField(max_length=100, verbose_name='Notice Period', blank=True, null=True)
    
    # File fields for resume and cover letter
    resume = models.FileField(upload_to='resumes/')
    cover_letter = models.FileField(upload_to='cover_letters/', blank=True, null=True)
    
    linkedin_url = models.URLField(blank=True, null=True)
    
    # Track the application status
    STATUS_CHOICES = [
        ('Pending', 'Pending Review'),
        ('Reviewed', 'Reviewed'),
        ('Interviewing', 'Interviewing'),
        ('Rejected', 'Rejected'),
        ('Hired', 'Hired'),
    ]
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='Pending')

    date_applied = models.DateTimeField(auto_now_add=True)

    # A generic JSON field to handle any additional fields added by the user
    additional_data = models.JSONField(default=dict, blank=True)
    
    def __str__(self):
        return f"Application from {self.first_name} {self.last_name} for {self.career.title}"

def document_upload_path(instance, filename):
    folder_name = instance.folder.name
    # os.path.join handles path construction for different operating systems
    return os.path.join('documents', folder_name, filename)

class Folder(models.Model):
    # Link each folder to a specific user
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='folders')
    name = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)
    
    # Add a unique constraint for the folder name per user
    class Meta:
        unique_together = ('user', 'name')
    
    def __str__(self):
        return self.name

class Document(models.Model):
    # Link each document to a specific user
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    
    # This is the foreign key relationship to Folder
    folder = models.ForeignKey(Folder, on_delete=models.CASCADE, related_name='documents')
    
    file = models.FileField(upload_to=document_upload_path)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    
    # Add this property to return only the file name
    @property
    def file_name_only(self):
        return os.path.basename(self.file.name)
    
    def __str__(self):
        return self.file.name


class CareerAdvanceAnalysis(models.Model):
    # Link to the incoming application
    application = models.ForeignKey(Apply_career, on_delete=models.CASCADE, related_name='advanced_analysis')
    
    # Fields from the JSON response
    candidate_name = models.CharField(max_length=255, blank=True)
    role_evaluated = models.CharField(max_length=255, blank=True)
    summary_verdict = models.CharField(max_length=50, blank=True)
    rationale = models.TextField(blank=True)
    
    # Gap Analysis - Stored as JSONField for flexibility
    gap_analysis = models.JSONField(default=list)
    
    # Stability Summary - Stored as JSONField
    stability_summary = models.JSONField(default=dict)
    
    # Scorecard - Stored as JSONField
    scorecard = models.JSONField(default=dict)
    
    # Alerts - Stored as JSONField
    alerts = models.JSONField(default=dict)
    
    # Bench Decision - Stored as JSONField
    bench_decision = models.JSONField(default=dict)

    # You can add a timestamp to track when the analysis was performed
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Advanced Analysis for {self.candidate_name} on {self.created_at.strftime('%Y-%m-%d')}"

    class Meta:
        # To ensure only one advanced analysis per application
        unique_together = ('application',)
        db_table = 'career_advance_analysis' # Use the specified table name

class CandidateAnalysis(models.Model):
    # Link to the incoming application (only one analysis per application)
    application = models.OneToOneField(
    Apply_career,
    on_delete=models.CASCADE,
    null=True,  # allow NULL
    blank=True  
    )
    # application = models.ForeignKey(
    #     Apply_career,
    #     on_delete=models.CASCADE,
    #     related_name="analyses"
    # )
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=255, null=True, blank=True)
    job_role = models.CharField(max_length=255, null=True, blank=True)
    phone_no = models.CharField(max_length=50, null=True, blank=True)
    hiring_recommendation = models.CharField(max_length=50, null=True, blank=True)
    suggested_salary_range = models.CharField(max_length=100, null=True, blank=True)
    interview_questions = models.TextField(null=True, blank=True)  # Stores JSON string

    analysis_summary = models.TextField(null=True, blank=True)  # Stores JSON string

    experience_match = models.CharField(max_length=50, null=True, blank=True)
    overall_experience = models.CharField(max_length=50, null=True, blank=True)
    current_company_name = models.CharField(max_length=255, null=True, blank=True)
    current_company_address = models.CharField(max_length=255, null=True, blank=True)

    fitment_verdict = models.CharField(max_length=50, null=True, blank=True)
    aggregate_score = models.CharField(max_length=50, null=True, blank=True)

    DECISION_CHOICES = [
        ('shortlisted', 'Shortlisted'),
        ('selected', 'Selected'),
        ('not_selected', 'Not Selected'),
        ('pending', 'Pending'),
    ]

    final_decision = models.CharField(
        max_length=20,
        choices=DECISION_CHOICES,
        default='pending',
        blank=True,
        null=True
    )

    final_salary = models.DecimalField(
        max_digits=10,
        decimal_places=2,
        blank=True,
        null=True
    )

    strategic_alignment = models.TextField(null=True, blank=True)
    quantifiable_impact = models.TextField(null=True, blank=True)
    potential_gaps_risks = models.TextField(null=True, blank=True)
    comparable_experience = models.TextField(null=True, blank=True)

    scoring_matrix_json = models.TextField(null=True, blank=True)
    bench_recommendation_json = models.TextField(null=True, blank=True)
    alternative_role_recommendations_json = models.TextField(null=True, blank=True)
    automated_recruiter_insights_json = models.TextField(null=True, blank=True)

    ai_summary = models.TextField(blank=True, null=True)
    confidence_score = models.IntegerField(blank=True, null=True)
    suggested_questions = models.TextField(blank=True, null=True)

    candidate_overview = models.TextField(null=True, blank=True)
    technical_prowess_json = models.TextField(null=True, blank=True)
    project_impact_json = models.TextField(null=True, blank=True)
    education_certifications_json = models.TextField(null=True, blank=True)
    overall_rating_summary = models.CharField(max_length=50, null=True, blank=True)
    conclusion_summary = models.TextField(null=True, blank=True)
    bland_call_id = models.CharField(max_length=100, blank=True, null=True)
    interview_status = models.CharField(max_length=50, null=True, blank=True, default='Pending')
    resume_file_path = models.CharField(max_length=255, null=True, blank=True)

    ANALYSIS_TYPES = (
        ('Manual', 'Manual ATS Analysis'),
        ('Basic', 'Basic ATS Analysis'),
        ('Advance', 'Advanced ATS Analysis'),
    )
    analysis_type = models.CharField(
        max_length=10,
        choices=ANALYSIS_TYPES,
        default='Manual',
        help_text="Type of resume analysis (Basic or Advanced)."
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        """Always return a safe string for admin display"""
        if self.full_name:
            return self.full_name
        if hasattr(self.application, "full_name") and self.application.full_name:
            return self.application.full_name
        return f"CandidateAnalysis #{self.pk}"
    


class Category(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='categories')
    ICON_CHOICES = [
        # Existing / basic
        ('fa-envelope-open-text', 'Marketing'),
        ('fa-headset', 'Customer Service'),
        ('fa-user-tie', 'Human Resource'),
        ('fa-tasks', 'Project Management'),
        ('fa-chart-line', 'Business Development'),
        ('fa-handshake', 'Sales & Communication'),
        ('fa-book-open', 'Teaching & Education'),
        ('fa-pencil-ruler', 'Design & Creative'),
        ('fa-laptop-code', 'Software Development'),
        ('fa-code', 'Frontend Development'),
        ('fa-server', 'Backend Development'),
        ('fa-database', 'Database Development'),
        ('fa-mobile-alt', 'Mobile App Development'),
        ('fa-cloud', 'Cloud / DevOps'),
        
        # ATS-specific additions
        ('fa-briefcase', 'Job Posting / Requisition'),
        ('fa-file-upload', 'Application Submission'),
        ('fa-file-alt', 'Resume / CV Processing'),
        ('fa-tasks', 'Pre-Screening / Assessment'),
        ('fa-calendar-alt', 'Interview Scheduling'),
        ('fa-comments', 'Interview Feedback'),
        ('fa-file-signature', 'Offer / Contract'),
        ('fa-user-plus', 'Onboarding'),
        ('fa-times-circle', 'Rejection'),
        ('fa-user-clock', 'Candidate Status Tracking'),
        ('fa-users', 'Internal / External Candidates'),
        ('fa-folder-open', 'Documents & Files'),
        ('fa-sticky-note', 'Notes & Comments'),
        ('fa-share-alt', 'Collaboration / Sharing'),
        ('fa-shield-alt', 'Background & Verification'),
        ('fa-chart-pie', 'Analytics & Reports'),
    ]

    name = models.CharField(max_length=100)
    vacancy_count = models.IntegerField(default=0)
    icon_class = models.CharField(
        max_length=50,
        choices=ICON_CHOICES,
        default='fa-tasks'
    ) 

    def __str__(self):
        return self.name

class CareerJob(models.Model):
    JOB_TYPES = [
        ('FT', 'Full Time'),
        ('PT', 'Part Time'),
        ('FE', 'Featured'),
    ]

    title = models.CharField(max_length=200)
    location = models.CharField(max_length=100)
    job_type = models.CharField(max_length=2, choices=JOB_TYPES)
    salary_range = models.CharField(max_length=50, default='$123 - $456')
    date_line = models.DateField()
    company_logo = models.ImageField(upload_to='company_logos/')
    category = models.ForeignKey(Category, on_delete=models.CASCADE)

    def __str__(self):
        return self.title
    

class JobApplicationFormSettings(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, unique=True)
    # Personal Details
    first_name_enabled = models.BooleanField(default=True)
    last_name_enabled = models.BooleanField(default=True)
    email_enabled = models.BooleanField(default=True)
    phone_enabled = models.BooleanField(default=True)

    # Professional Details
    experience_enabled = models.BooleanField(default=True)
    current_ctc_enabled = models.BooleanField(default=True)
    expected_ctc_enabled = models.BooleanField(default=True)
    notice_period_enabled = models.BooleanField(default=True)
    qualification_enabled = models.BooleanField(default=True)
    linkedin_url_enabled = models.BooleanField(default=True)

    # Documents
    resume_enabled = models.BooleanField(default=True)
    cover_letter_enabled = models.BooleanField(default=True)
    
    # A generic JSON field to handle any additional fields added by the user
    additional_fields = models.JSONField(default=list)
    
    def __str__(self):
        return f"Settings for {self.user.username}"