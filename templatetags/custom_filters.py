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