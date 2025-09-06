from django.core.mail import send_mail
from django.conf import settings
from .models import EmailConfiguration

def send_configured_email(subject, message, recipient_list, from_email=None):
    """
    Sends an email using the configuration settings stored in the database.
    """
    try:
        # Load the email configuration from the database.
        config = EmailConfiguration.load()
        
        # Use the configured settings to get a connection.
        connection = config.get_connection()
        
        # Determine the 'from' email address.
        if from_email is None:
            from_email = config.email_from if config.email_from else config.email_host_user
        
        # Send the email with the dynamic connection.
        send_mail(
            subject=subject,
            message=message,
            from_email=from_email,
            recipient_list=recipient_list,
            connection=connection,
            fail_silently=False,
        )
        print("Email sent successfully using dynamic configuration.")
        return True

    except EmailConfiguration.DoesNotExist:
        print("Error: EmailConfiguration not found in the database. Please create a configuration entry.")
        return False
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
