from django import template

register = template.Library()

@register.filter
def replace(value, arg):
    """
    Replaces all occurrences of a substring with another substring in a string.
    Usage: {{ some_string|replace:"old_substring,new_substring" }}
    """
    try:
        old, new = arg.split(',')
        return value.replace(old, new)
    except ValueError:
        return value # Return original value if arg is not in 'old,new' format

@register.filter
def replace_underscore_with_space(value):
    """
    Replaces underscores with spaces in a string.
    Usage: {{ some_string|replace_underscore_with_space }}
    """
    return value.replace('_', ' ')




###################
import re
from django import template

register = template.Library()

@register.filter(name='to_lpa')
def convert_to_lpa(salary_string):
    """
    Converts a salary string (e.g., '₹50,000 - ₹65,000 per month') to LPA.
    Handles 'monthly' and 'LPA' formats.
    """
    if not salary_string:
        return "N/A"

    salary_string = salary_string.lower().replace(',', '')
    numbers = re.findall(r'\d+', salary_string)
    
    if not numbers:
        return "N/A"
    
    if 'lpa' in salary_string or 'annum' in salary_string:
        # Already in LPA, just clean and format
        min_salary = int(numbers[0])
        max_salary = int(numbers[1]) if len(numbers) > 1 else min_salary
        return f"₹{min_salary / 100000:.2f} - ₹{max_salary / 100000:.2f} LPA"
    
    if 'month' in salary_string or 'monthly' in salary_string:
        # Convert monthly to LPA
        min_monthly = int(numbers[0])
        min_lpa = (min_monthly * 12) / 100000
        
        max_lpa = None
        if len(numbers) > 1:
            max_monthly = int(numbers[1])
            max_lpa = (max_monthly * 12) / 100000
        
        if max_lpa:
            return f"₹{min_lpa:.2f} - ₹{max_lpa:.2f} LPA"
        else:
            return f"₹{min_lpa:.2f} LPA"

    # Default to LPA conversion if no unit is specified
    min_salary = int(numbers[0])
    min_lpa = min_salary / 100000

    if len(numbers) > 1:
        max_salary = int(numbers[1])
        max_lpa = max_salary / 100000
        return f"₹{min_lpa:.2f} - ₹{max_lpa:.2f} LPA"
    else:
        return f"₹{min_lpa:.2f} LPA"