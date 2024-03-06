import os
import base64
import google.auth
from google.auth.transport.requests import Request
from google.oauth2 import service_account
from googleapiclient.discovery import build

# Load credentials from JSON file
credentials_file = 'credentials.json'
credentials = service_account.Credentials.from_service_account_file(
    credentials_file, scopes=['https://www.googleapis.com/auth/gmail.compose']
)

# Build the Gmail API service
service = build('gmail', 'v1', credentials=credentials)

# Create a draft email
def create_draft(sender, to, subject, message):
    email = f"""\
    From: {sender}
    To: {to}
    Subject: {subject}

    {message}
    """

    draft = {
        'message': {
            'raw': base64.urlsafe_b64encode(email.encode()).decode()
        }
    }

    draft = service.users().drafts().create(userId='me', body=draft).execute()
    return draft

# Example usage
sender = 'your-email@gmail.com'
to = 'recipient@example.com'
subject = 'Hello from Python!'
message = 'This is a test email created using the Gmail API'

draft = create_draft(sender, to, subject, message)
print(f"Draft created with ID: {draft['id']}")