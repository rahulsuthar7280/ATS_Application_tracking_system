# # myhrproject/hr_app/forms.py

# from django import forms
# from wtforms.validators import ValidationError # This line is from Flask-WTF, remove it if not used elsewhere
# import re

# class ResumeUploadForm(forms.Form):
#     """
#     Form for uploading a resume PDF and specifying job role/experience.
#     """
#     resume_pdf = forms.FileField(
#         label="Upload a PDF Resume",
#         help_text="Only PDF files are allowed.",
#         widget=forms.FileInput(attrs={'accept': '.pdf'})
#     )
#     job_role = forms.CharField(
#         label="Job Role",
#         max_length=255,
#         help_text="e.g., 'Senior Python Developer', 'Data Scientist'"
#     )
    
#     # Experience Level fields
#     TARGET_EXPERIENCE_CHOICES = [
#         ("Any Experience Level", "Any Experience Level"),
#         ("Specific Range (Years)", "Specific Range (Years)"),
#         ("Minimum Years Required", "Minimum Years Required"),
#     ]
#     target_experience_type = forms.ChoiceField(
#         label="Target Experience Level",
#         choices=TARGET_EXPERIENCE_CHOICES,
#         initial="Any Experience Level",
#         # Removed onchange submit as it's not ideal for Django forms.
#         # JavaScript will handle dynamic display without form submission.
#     )
#     min_years_required = forms.IntegerField(
#         label="Min Years",
#         min_value=0,
#         max_value=30,
#         required=False, # Only required if specific experience type is chosen
#         initial=0
#     )
#     max_years_required = forms.IntegerField(
#         label="Max Years",
#         min_value=0,
#         max_value=30,
#         required=False, # Only required if specific experience type is chosen
#         initial=0
#     )

#     def clean(self):
#         cleaned_data = super().clean()
#         target_type = cleaned_data.get('target_experience_type')
#         min_years = cleaned_data.get('min_years_required')
#         max_years = cleaned_data.get('max_years_required')

#         if target_type == "Specific Range (Years)":
#             if min_years is None or max_years is None:
#                 self.add_error('min_years_required', "Minimum and Maximum years are required for 'Specific Range'.")
#                 # self.add_error('max_years_required', "Minimum and Maximum years are required for 'Specific Range'.") # Redundant error
#             elif min_years > max_years:
#                 self.add_error('min_years_required', "Min Years cannot be greater than Max Years.")
#         elif target_type == "Minimum Years Required":
#             if min_years is None:
#                 self.add_error('min_years_required', "Minimum years is required for 'Minimum Years Required'.")
#             cleaned_data['max_years_required'] = 0 # Ensure max is 0 if not used
#         else: # Any Experience Level
#             cleaned_data['min_years_required'] = 0
#             cleaned_data['max_years_required'] = 0
        
#         return cleaned_data

# # ... (other forms like FinalDecisionForm, PhoneNumberForm) ...
# # Ensure you have the rest of your forms here if they exist.

# class FinalDecisionForm(forms.Form):
#     """
#     Form for confirming the final hiring decision and salary.
#     """
#     FINAL_DECISION_CHOICES = [
#         ("Selected", "Selected"),
#         ("Not Selected", "Not Selected"),
#     ]
#     final_decision = forms.ChoiceField(
#         label="Confirm hiring decision:",
#         choices=FINAL_DECISION_CHOICES,
#         widget=forms.RadioSelect
#     )
#     final_salary = forms.DecimalField(
#         label="Final Fixed Salary (₹ per month)",
#         min_value=0,
#         decimal_places=2,
#         required=False # Can be left blank if not selected
#     )

# class PhoneNumberForm(forms.Form):
#     """
#     Simple form to allow manual phone number input for interview if not found in resume.
#     """
#     phone_number = forms.CharField(
#         label="Phone Number for Interview",
#         max_length=20,
#         help_text="e.g., +919876543210",
#         required=False
#     )

#     def clean_phone_number(self):
#         phone_number = self.cleaned_data.get('phone_number')
#         if phone_number and not self.is_valid_phone_number(phone_number):
#             raise forms.ValidationError("Phone number must start with '+' and country code, e.g., +919876543210.")
#         return phone_number

#     def is_valid_phone_number(self, phone_number):
#         import re
#         pattern = re.compile(r'^\+\d{1,3}\d{7,15}$')
#         return bool(pattern.match(phone_number))



# myhrproject/hr_app/forms.py

from django import forms
import re # Keep re for phone number validation if needed elsewhere
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from .models import User
# Custom widgets to apply the 'form-control-custom' class for consistent styling
class CustomTextInput(forms.TextInput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs.update({'class': 'form-control'})

class CustomSelect(forms.Select):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs.update({'class': 'form-control'})

# For file inputs, we'll use 'form-control' as it's Bootstrap's default
# for file types and looks good.
class CustomFileInput(forms.FileInput):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attrs.update({'class': 'form-control'}) # Using Bootstrap's default file input class


class ResumeUploadForm(forms.Form):
    """
    Form for uploading a resume PDF and specifying job role/experience.
    """
    resume_pdf = forms.FileField(
        label="Upload a PDF Resume",
        help_text="Only PDF files are allowed.",
        widget=CustomFileInput(attrs={'accept': '.pdf'}) # Using CustomFileInput
    )
    
    # NEW: Job Description File Field
    job_description = forms.FileField(
        label='Upload Job Description (Optional)',
        required=False, # This field is optional
        widget=CustomFileInput(attrs={'accept': '.pdf,.doc,.docx,.txt'}), # Accepts various document types
        help_text="Accepts PDF, DOCX, or plain text files."
    )

    job_role = forms.CharField(
        label="Job Role",
        max_length=255,
        help_text="e.g., 'Senior Python Developer', 'Data Scientist'",
        widget=CustomTextInput() # Using CustomTextInput
    )
    
    # Experience Level fields
    TARGET_EXPERIENCE_CHOICES = [
        ("Any Experience Level", "Any Experience Level"),
        ("Specific Range (Years)", "Specific Range (Years)"),
        ("Minimum Years Required", "Minimum Years Required"),
    ]
    target_experience_type = forms.ChoiceField(
        label="Target Experience Level",
        choices=TARGET_EXPERIENCE_CHOICES,
        initial="Any Experience Level",
        widget=CustomSelect() # Using CustomSelect
    )
    min_years_required = forms.IntegerField(
        label="Min Years",
        min_value=0,
        max_value=30,
        required=False, # Only required if specific experience type is chosen
        initial=0,
        widget=CustomTextInput() # Using CustomTextInput
    )
    max_years_required = forms.IntegerField(
        label="Max Years",
        min_value=0,
        max_value=30,
        required=False, # Only required if specific experience type is chosen
        initial=0,
        widget=CustomTextInput() # Using CustomTextInput
    )

    def clean(self):
        cleaned_data = super().clean()
        target_type = cleaned_data.get('target_experience_type')
        min_years = cleaned_data.get('min_years_required')
        max_years = cleaned_data.get('max_years_required')

        if target_type == "Specific Range (Years)":
            if min_years is None or max_years is None:
                self.add_error('min_years_required', "Minimum and Maximum years are required for 'Specific Range'.")
            elif min_years > max_years:
                self.add_error('min_years_required', "Min Years cannot be greater than Max Years.")
        elif target_type == "Minimum Years Required":
            if min_years is None:
                self.add_error('min_years_required', "Minimum years is required for 'Minimum Years Required'.")
            cleaned_data['max_years_required'] = 0 # Ensure max is 0 if not used
        else: # Any Experience Level
            cleaned_data['min_years_required'] = 0
            cleaned_data['max_years_required'] = 0
        
        return cleaned_data

# Ensure you have the rest of your forms here if they exist.
class FinalDecisionForm(forms.Form):
    """
    Form for confirming the final hiring decision and salary.
    """
    FINAL_DECISION_CHOICES = [
        ("selected", "Selected"), # Corrected value
        ("not_selected", "Not Selected"), # Corrected value
        ("shortlisted", "Shortlisted"), # Added option
    ]
    final_decision = forms.ChoiceField(
        label="Confirm hiring decision:",
        choices=FINAL_DECISION_CHOICES,
        widget=forms.RadioSelect
    )
    final_salary = forms.DecimalField(
        label="Final Fixed Salary (₹ per month)",
        min_value=0,
        decimal_places=2,
        required=False
    )

class PhoneNumberForm(forms.Form):
    """
    Simple form to allow manual phone number input for interview if not found in resume.
    """
    phone_number = forms.CharField(
        label="Phone Number for Interview",
        max_length=20,
        help_text="e.g., +919876543210",
        required=False,
        widget=CustomTextInput() # Using CustomTextInput
    )

    def clean_phone_number(self):
        phone_number = self.cleaned_data.get('phone_number')
        if phone_number and not self.is_valid_phone_number(phone_number):
            raise forms.ValidationError("Phone number must start with '+' and country code, e.g., +919876543210.")
        return phone_number

    def is_valid_phone_number(self, phone_number):
        # Using re imported at the top of the file
        pattern = re.compile(r'^\+\d{1,3}\d{7,15}$')
        return bool(pattern.match(phone_number))


class CustomUserCreationForm(UserCreationForm):
    """
    A custom form for creating new users, including the 'role' field.
    """
    # The 'role' field will be automatically included because we're extending UserCreationForm
    # and our User model has the 'role' field.
    # However, by default, UserCreationForm only shows username and password.
    # We need to explicitly add 'role' to the fields.
    class Meta(UserCreationForm.Meta):
        model = User
        fields = UserCreationForm.Meta.fields + ('role',) # Add 'role' to the fields

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Make 'role' field visible and allow selection during signup
        # For a public signup, you might want to hide this and set a default 'user' role
        # or only show it to superadmins creating new users.
        # For this example, we'll make it selectable.
        self.fields['role'].required = True # Make role selection mandatory during signup

class CustomAuthenticationForm(AuthenticationForm):
    """
    A custom authentication form. Django's default AuthenticationForm is usually
    sufficient, but you can customize it here if needed (e.g., adding a remember me checkbox).
    """
    class Meta:
        model = User
        fields = ['username', 'password'] # Standard fields for login


# myhrproject/hr_app/forms.py (Create this file if it doesn't exist)

# from django import forms
# from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
# from .models import User, CandidateAnalysis # Import your custom User model

# class CustomUserCreationForm(UserCreationForm):
#     """
#     A custom form for creating new users, including the 'role' field.
#     """
#     # The 'role' field will be automatically included because we're extending UserCreationForm
#     # and our User model has the 'role' field.
#     # However, by default, UserCreationForm only shows username and password.
#     # We need to explicitly add 'role' to the fields.
#     class Meta(UserCreationForm.Meta):
#         model = User
#         fields = UserCreationForm.Meta.fields + ('role',) # Add 'role' to the fields

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Make 'role' field visible and allow selection during signup
#         # For a public signup, you might want to hide this and set a default 'user' role
#         # or only show it to superadmins creating new users.
#         # For this example, we'll make it selectable.
#         self.fields['role'].required = True # Make role selection mandatory during signup

# class CustomAuthenticationForm(AuthenticationForm):
#     """
#     A custom authentication form. Django's default AuthenticationForm is usually
#     sufficient, but you can customize it here if needed (e.g., adding a remember me checkbox).
#     """
#     class Meta:
#         model = User
#         fields = ['username', 'password'] # Standard fields for login

# # Your existing forms (from views.py snippet, assuming they are in forms.py)
# class ResumeUploadForm(forms.Form):
#     resume_pdf = forms.FileField(label="Upload Resume (PDF)", help_text="Upload the candidate's resume in PDF format.")
#     job_description = forms.FileField(label="Upload Job Description (PDF/TXT)", help_text="Upload the job description in PDF or TXT format.", required=False)
#     job_role = forms.CharField(max_length=255, label="Target Job Role", help_text="e.g., 'Software Engineer', 'Data Analyst'")
    
#     TARGET_EXPERIENCE_CHOICES = [
#         ('Any', 'Any'),
#         ('Specific Range (Years)', 'Specific Range (Years)'),
#         ('Minimum Years Required', 'Minimum Years Required'),
#     ]
#     target_experience_type = forms.ChoiceField(
#         choices=TARGET_EXPERIENCE_CHOICES,
#         label="Target Experience Type",
#         initial='Any'
#     )
#     min_years_required = forms.IntegerField(
#         label="Minimum Years Required",
#         required=False,
#         min_value=0,
#         help_text="Required if 'Specific Range' or 'Minimum Years' is selected."
#     )
#     max_years_required = forms.IntegerField(
#         label="Maximum Years Required",
#         required=False,
#         min_value=0,
#         help_text="Required if 'Specific Range' is selected."
#     )

#     def clean(self):
#         cleaned_data = super().clean()
#         target_type = cleaned_data.get('target_experience_type')
#         min_years = cleaned_data.get('min_years_required')
#         max_years = cleaned_data.get('max_years_required')

#         if target_type == 'Specific Range (Years)':
#             if min_years is None:
#                 self.add_error('min_years_required', "Minimum years is required for 'Specific Range'.")
#             if max_years is None:
#                 self.add_error('max_years_required', "Maximum years is required for 'Specific Range'.")
#             if min_years is not None and max_years is not None and min_years > max_years:
#                 self.add_error('max_years_required', "Maximum years must be greater than or equal to minimum years.")
#         elif target_type == 'Minimum Years Required':
#             if min_years is None:
#                 self.add_error('min_years_required', "Minimum years is required for 'Minimum Years Required'.")
        
#         return cleaned_data

# class FinalDecisionForm(forms.ModelForm):
#     class Meta:
#         model = CandidateAnalysis
#         fields = ['final_decision', 'final_salary']
#         widgets = {
#             'final_decision': forms.Select(choices=[
#                 ('', 'Select Decision'), # Empty choice
#                 ('Selected', 'Selected'),
#                 ('Not Selected', 'Not Selected'),
#             ]),
#             'final_salary': forms.TextInput(attrs={'placeholder': 'e.g., 60000 USD/year'}),
#         }

# class PhoneNumberForm(forms.Form):
#     phone_number = forms.CharField(
#         max_length=20,
#         required=True,
#         help_text="Enter phone number with country code, e.g., +12345678900"
#     )

#     def is_valid_phone_number(self, phone_number):
#         # Basic validation: starts with '+' and contains only digits after that
#         return phone_number.startswith('+') and phone_number[1:].isdigit()

#     def clean_phone_number(self):
#         phone_number = self.cleaned_data['phone_number']
#         if not self.is_valid_phone_number(phone_number):
#             raise forms.ValidationError("Phone number must start with '+' and contain only digits after the plus sign.")
#         return phone_number

