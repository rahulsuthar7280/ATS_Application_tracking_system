# linkedin_multi_poster.py
# This script automates the process of posting multiple jobs on LinkedIn
# using Selenium and a list of job description dictionaries.
# Note: LinkedIn's UI may change, which could break element locators.
# You will need to inspect the page and update them if needed.

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import Select
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json

# --- LinkedIn Credentials ---
# REPLACE these with your actual LinkedIn email and password.
# It is highly recommended to use environment variables for production.
LINKEDIN_USERNAME = "rahulsuthar7280@gmail.com"
LINKEDIN_PASSWORD = "rahul@7280"
# --------------------------

def post_jobs_to_linkedin(jobs_data, username, password):
    """
    Automates logging into LinkedIn and posting a list of jobs.

    Args:
        jobs_data (list of dict): A list of dictionaries containing job details.
        username (str): Your LinkedIn email or username.
        password (str): Your LinkedIn password.
    """
    if not jobs_data:
        print("No job data provided. Exiting.")
        return

    try:
        service = ChromeService(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service)
        wait = WebDriverWait(driver, 20)

        # 1. Navigate to LinkedIn login page and log in
        print("Navigating to LinkedIn login page...")
        driver.get("https://www.linkedin.com/login")
        driver.maximize_window()
        time.sleep(2)

        print("Logging in...")
        username_field = wait.until(EC.presence_of_element_located((By.ID, "username")))
        password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
        username_field.send_keys(username)
        password_field.send_keys(password)

        # Updated: Using a more stable XPath or CSS selector for the login button
        # This will find the button by its type='submit' and a common class.
        sign_in_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
        sign_in_button.click()
        time.sleep(5)

        if "feed" not in driver.current_url:
            print("Login failed. Check your credentials.")
            return

        print("Login successful. Starting job posting process...")

        for i, job_data in enumerate(jobs_data):
            print(f"\n--- Posting Job {i + 1}/{len(jobs_data)}: {job_data.get('title', 'Untitled')} ---")
            
            # Navigate to the job posting page for each job
            driver.get("https://www.linkedin.com/talent/post-a-job")
            wait.until(EC.url_contains("post-a-job"))

            # 2. Fill out the job title and click "Post for free"
            print("Filling out job title...")
            job_title_input = wait.until(EC.presence_of_element_located((By.ID, "job-title-input")))
            job_title_input.send_keys(job_data.get('title', ''))
            time.sleep(1) # Added a small delay to ensure the button is fully rendered

            print("Attempting to click 'Post for free'...")
            # Use explicit wait until job title input is no longer present,
            # indicating a page transition or element change.
            try:
                # First, check if the input is still there
                wait.until(EC.invisibility_of_element_located((By.ID, "job-title-input")))
            except:
                print("job-title-input field is still present after typing. Proceeding with click.")
            
            # Updated to find the button that contains the span with the specific text,
            # ensuring we click the parent button which is the actual interactive element.
            post_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//button[./span[text()='Post for free']]")))
            
            # Using JavaScript to click, as it's often more reliable for stubborn buttons
            driver.execute_script("arguments[0].click();", post_button)

            # Wait for the next page to load
            wait.until(EC.url_contains("talent/job-post"))
            time.sleep(3)

            # 3. Fill out the job details form
            print("Filling out job details...")

            # Fill Location (City, State, Country)
            location_input = wait.until(EC.presence_of_element_located((By.ID, "job-location-input")))
            location_full = f"{job_data.get('city', '')}, {job_data.get('state', '')}, {job_data.get('country', '')}"
            location_input.send_keys(location_full)
            time.sleep(2)
            location_input.send_keys(Keys.DOWN)
            location_input.send_keys(Keys.RETURN)
            
            # Fill Job Description
            full_description = f"""
            {job_data.get('overview', '')}

            Responsibilities:
            {job_data.get('responsibilities', '')}

            Required Skills:
            {job_data.get('required_skills', '')}
            
            Preferred Skills:
            {job_data.get('preferred_skills', '')}
            
            Education & Experience:
            {job_data.get('education_experience', '')}
            
            Benefits:
            {job_data.get('benefits', '')}
            """
            
            description_field = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "div[aria-label='Job description']")))
            description_field.send_keys(full_description)
            
            # Select Employment Type
            employment_type_select_element = wait.until(EC.presence_of_element_located((By.ID, "employment-type-select")))
            select = Select(employment_type_select_element)
            select.select_by_value(job_data.get('employment_type', 'full-time'))

            # Select Job Level
            job_level_select_element = wait.until(EC.presence_of_element_located((By.ID, "job-level-select")))
            select = Select(job_level_select_element)
            select.select_by_value(job_data.get('job_level', 'mid'))
            
            # Scroll down to see the next sections
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)

            # 4. Click the "Next" button to proceed
            print("Submitting the job details...")
            next_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='next-button']")))
            driver.execute_script("arguments[0].click();", next_button)
            time.sleep(5)
            
            # 5. Handle the "How would you like to receive applicants?" screen
            print("Selecting application method...")
            # We'll assume the most common option: 'Apply with your resume' which is 'External Website' or similar.
            # Look for the radio button for 'Email' or 'Recruit'. This selector might need to be adjusted.
            try:
                # Find the radio button for receiving applicants via email
                radio_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[label[span[text()='Email']]]//input[@type='radio']")))
                driver.execute_script("arguments[0].click();", radio_button)
                time.sleep(2)
                
                # Enter email address (this is a placeholder, you'd need to add the email to your job data)
                email_input = wait.until(EC.presence_of_element_located((By.ID, "email-input")))
                email_input.send_keys("recruiting@example.com") # Placeholder email
                
            except Exception as e:
                print(f"Could not find or click the 'Email' radio button. Trying 'LinkedIn Recruiter'...")
                # Fallback to LinkedIn Recruiter if 'Email' isn't an option
                try:
                    radio_button = wait.until(EC.element_to_be_clickable((By.XPATH, "//div[label[span[text()='LinkedIn Recruiter']]]//input[@type='radio']")))
                    driver.execute_script("arguments[0].click();", radio_button)
                except Exception as e:
                    print(f"Failed to select a valid application method: {e}")
                    # You might need to add more robust handling or manually check the UI.

            # 6. Click the "Next" button on the applicants screen
            print("Submitting applicant details...")
            next_button_applicants = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='next-button']")))
            driver.execute_script("arguments[0].click();", next_button_applicants)
            time.sleep(5)

            # 7. Final Review and Publish
            print("Reviewing and publishing job...")
            publish_button = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-testid='publish-button']")))
            driver.execute_script("arguments[0].click();", publish_button)
            
            print(f"Job {i + 1} '{job_data.get('title', '')}' published successfully!")
            
            # Wait for a success message or for the page to navigate away
            time.sleep(10)

    except Exception as e:
        print(f"An error occurred during the job posting process: {e}")
        # The browser will stay open on error for debugging purposes.
        input("Press Enter to close the browser...")

    finally:
        pass
