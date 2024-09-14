import re

def extract_data(text):
    # Define regex patterns for emails, phone numbers, dates, and URLs
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b'
    phone_pattern = r'\+?\d[\d\s()-]{8,}\d'
    date_pattern = r'\b(?:\d{2}[-/]\d{2}[-/]\d{4}|\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{2,4}|(?:\d{2}[-/]){2}\d{4}|(?:\d{4}[-/]){2}\d{2})\b'
    url_pattern = r'\b(?:http|https|ftp)://[^\s/$.?#].[^\s]*\b'

    # Use regex to find all matches
    emails = re.findall(email_pattern, text, re.IGNORECASE)
    phones = re.findall(phone_pattern, text)
    dates = re.findall(date_pattern, text)
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    
    # Return the results as a dictionary with lists
    return {
        "emails": emails,
        "phones": phones,
        "dates": dates,
        "urls": urls
    }

# Read data from 'data.txt' with UTF-8 encoding
with open('data.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Extract data using regex
extracted_data = extract_data(text)

# Count occurrences of each type
count_emails = len(extracted_data["emails"])
count_phones = len(extracted_data["phones"])
count_dates = len(extracted_data["dates"])
count_urls = len(extracted_data["urls"])

# Print counts
print(f"Count from regex extraction:")
print(f"Emails: {count_emails}")
print(f"Phones: {count_phones}")
print(f"Dates: {count_dates}")
print(f"URLs: {count_urls}")

predefined_counts = {
    "emails": 11,  
    "phones": 13,  
    "dates": 13,   
    "urls": 12     # Replace with the actual number of URLs
}

# Print predefined counts
print("\nPredefined counts:")
print(f"Emails: {predefined_counts['emails']}")
print(f"Phones: {predefined_counts['phones']}")
print(f"Dates: {predefined_counts['dates']}")
print(f"URLs: {predefined_counts['urls']}")

# Compare counts
print("\nComparison results:")
print(f"Emails match: {count_emails == predefined_counts['emails']}")
print(f"Phones match: {count_phones == predefined_counts['phones']}")
print(f"Dates match: {count_dates == predefined_counts['dates']}")
print(f"URLs match: {count_urls == predefined_counts['urls']}")
