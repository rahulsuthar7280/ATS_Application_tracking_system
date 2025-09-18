# myhrproject/hr_app/urls.py

from django.urls import path
from hr_app import views
from django.conf.urls.static import static
from django.conf import settings

urlpatterns = [
    # Authentication URLs
    path('signup/', views.signup_view, name='signup'),
    path('signin/', views.signin_view, name='signin'),
    path('signout/', views.signout_view, name='signout'),


    path('create-user/', views.create_user, name='create_user'),


    # Role-based Dashboards (new)
    path('admin_dashboard/', views.admin_dashboard_view, name='admin_dashboard'),
    path('superadmin_dashboard/', views.superadmin_dashboard_view, name='superadmin_dashboard'),
    path('toggle_user_status/<int:user_id>/', views.toggle_user_status, name='toggle_user_status'),
    # path('reset-password/<int:user_id>/', views.reset_user_password, name='reset_user_password'),
    path('set-password/<int:user_id>/', views.set_user_password, name='set_user_password'),

    # Existing URLs (ensure they are protected with @login_required in views.py)
    path('', views.home, name='home'), # Redirects to dashboard now
    path('resume-analysis/', views.resume_analysis_view, name='resume_analysis'),
    path('basic_resume_analysis/', views.basic_resume_analysis_view, name='basic_resume_analysis'),
    path('advance_resume_analysis/', views.advance_resume_analysis_view, name='advance_resume_analysis'),
    path('initiate_call_interview/', views.initiate_call_interview, name='initiate_call_interview'),
    path('candidate_profile/', views.candidate_profile, name='candidate_profile'),
    path('interviews/', views.interview_dashboard_view, name='interviews'),
    path('interviews/<int:candidate_id>/', views.interview_detail_view, name='interview_detail'),

    # path('interview-status/', views.interview_status_view, name='interview_status'),
    path('interview-status/<int:candidate_id>/', views.interview_status_view, name='interview_status'),
    path('records/', views.candidate_records_view, name='records'),
    path('selected_candidate/', views.selected_candidate, name='selected_candidate'),
    path('rejected_candidate/', views.rejected_candidate, name='rejected_candidate'),
    path('shortlisted_candidate/', views.shortlisted_candidate, name='shortlisted_candidate'),
    path('airtable-data/', views.airtable_data_view, name='airtable_data'),
    path('post-to-airtable/', views.post_data_to_airtable_view, name='post_to_airtable'),
    path('recommendations/', views.top_recommendations_view, name='top_recommendations'),
    path('dashboard/', views.dashboard, name='dashboard'), # Your main dashboard
    path('candidate_profile/<int:candidate_id>/', views.candidate_profile_view, name='candidate_profile'),

    # NEW: URL for showing upcoming applications/mails
    # path('applications/', views.show_unread_emails, name='show_applications'),
    path("applications/", views.show_unread_emails, name="show_unread_emails"),
    path('update_application_data/', views.update_application_data, name='update_application_data'),

    # NEW: URL for processing ATS options
    path('process_ats/<str:email_id>/<str:ats_type>/', views.process_ats_option, name='process_ats_option'),

    path('configure-email/', views.configure_email, name='configure_email'),
    path('send-job-description/', views.send_job_description, name='send_job_description'),
    # path('success/', views.success_page, name='success_page'),
    # path('configure_email/', views.configure_email, name='configure_email'),
    # path('send_job_description/', views.send_job_description, name='send_job_description'),
    path('sent_emails/', views.sent_emails, name='sent_emails'),
    path('inbox/', views.inbox, name='inbox'),
    path('success/', views.success_page, name='success_page'),
    path('get_job_description_content/<int:job_id>/', views.get_job_description_content, name='get_job_description_content'),
    path('email_dashboard/', views.email_dashboard, name='email_dashboard'),


    path('analyze-resume/<int:email_id>/<str:analysis_type>/<int:job_description_id>/', views.analyze_resume, name='analyze_resume_with_jd'),
    path('analyze-resume/<int:email_id>/<str:analysis_type>/', views.analyze_resume, name='analyze_resume_without_jd'),

    path('job-descriptions/', views.all_job_descriptions, name='all_job_descriptions'),
    path('job-descriptions/create/', views.create_job_description, name='create_jd'),
    path('job-descriptions/upload/', views.upload_job_description, name='upload_jd'),
    path('job-descriptions/<int:jd_id>/edit/', views.edit_job_description, name='edit_jd'), # New Edit path
    path('delete-jd/<int:jd_id>/', views.delete_jd, name='delete_jd'),
    path('analyze-jd/<int:jd_id>/', views.analyze_jd, name='analyze_jd'),# You'll need to create analyze_jd.html

    path('analyze-resume/<int:email_id>/<str:analysis_type>/<int:jd_id>/', views.analyze_application_view, name='analyze_application_with_jd'),
    path('analyze-resume/<int:email_id>/<str:analysis_type>/', views.analyze_application_view, name='analyze_application'),
    path('analysis-results/<int:analysis_id>/', views.analysis_results_view, name='analysis_results'),
    # New URL for the HTML page
    
    # path('results/<uuid:email_id>/<str:analysis_type>/', views.analysis_page, name='analysis_page_no_jd'),
    # path('results/<uuid:email_id>/<str:analysis_type>/<uuid:job_description_id>/', views.analysis_page, name='analysis_page_with_jd'),
    path('calendar_scheduler/', views.calendar_scheduler, name='calendar_scheduler'),

    path('post-jobs/', views.post_jobs, name='post_jobs_view'),

    path('career_portal/', views.list_careers, name='career_portal'),
    path('toggle/<int:pk>/', views.toggle_status, name='toggle_status'),
    path('share/<int:pk>/', views.share_career, name='share_career'),
    path('careers/<int:pk>/', views.career_detail, name='career_detail'),

    # path('basic-ats/<int:application_id>/<int:job_description_id>/', views.basic_ats_analysis, name='basic_ats_analysis'),
    # path('basic-ats/<int:application_id>/', views.basic_ats_analysis, name='basic_ats_analysis'),
    # path('advance-ats/<int:application_id>/', views.advance_ats_analysis, name='advance_ats_analysis'),
    path('basic-ats/<int:application_id>/<int:job_description_id>/', views.basic_ats_analysis, name='basic_ats_analysis'),

    path('advance-ats/<int:application_id>/', views.advance_ats_analysis, name='advance_ats_analysis'),


    path('file_manager/', views.file_manager_view, name='file_manager'),
    path('browse/<int:folder_id>/', views.file_manager_view, name='browse_folder'),
    path('create_folder/', views.create_folder_view, name='create_folder'),
    path('delete_folder/<int:folder_id>/', views.delete_folder_view, name='delete_folder'),
    path('edit_folder/<int:folder_id>/', views.edit_folder_view, name='edit_folder'),
    path('delete_document/<int:document_id>/', views.delete_document_view, name='delete_document'),
    path('upload_file/', views.upload_file_view, name='upload_file'),
    path('ai_sort/', views.ai_auto_sort_view, name='ai_auto_sort'),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

    
