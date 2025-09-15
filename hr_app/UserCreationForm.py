from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import User

class AdminUserCreationForm(UserCreationForm):
    """
    A form that creates a user, but excludes the 'superadmin'
    option from the role choices.
    """
    class Meta:
        model = User
        fields = ('username', 'email', 'role',)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Exclude 'superadmin' from the role choices
        role_choices = list(User.ROLE_CHOICES)
        self.fields['role'].choices = [choice for choice in role_choices if choice[0] != 'superadmin']

    def save(self, commit=True):
        """
        Custom save method to handle setting the user's role and password.
        """
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password"])
        user.role = self.cleaned_data["role"]
        if commit:
            user.save()
        return user
