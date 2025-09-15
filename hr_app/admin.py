from django.contrib import admin

# Register your models here.
# from django.contrib import admin
from .models import CandidateAnalysis, User, Application, JobDescriptionDocument, EmailConfiguration, SentEmail, JobPosting, CareerPage, Apply_career


@admin.register(CandidateAnalysis)
class CandidateAnalysisAdmin(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]
    
@admin.register(User)
class User(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]
    
@admin.register(Application)
class Application(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]
    
@admin.register(JobDescriptionDocument)
class JobDescriptionDocument(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]

@admin.register(EmailConfiguration)
class EmailConfiguration(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]
    
@admin.register(SentEmail)
class SentEmail(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]
    

@admin.register(JobPosting)
class JobPosting(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]
    

@admin.register(CareerPage)
class CareerPage(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]
    
@admin.register(Apply_career)
class Apply_career(admin.ModelAdmin):
    def get_list_display(self, request):
        return [field.name for field in self.model._meta.fields]