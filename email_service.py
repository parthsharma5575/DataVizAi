import os
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail

def send_otp_email(to_email, otp):
    """Sends an OTP to the user's email address using SendGrid."""
    sendgrid_api_key = os.environ.get('SENDGRID_API_KEY')
    from_email = os.environ.get('FROM_EMAIL')

    if not sendgrid_api_key or not from_email or from_email == 'your-email@example.com':
        print("\n\n" + "="*50)
        print("WARNING: SendGrid not configured.")
        print("Please set SENDGRID_API_KEY and FROM_EMAIL in your .env file.")
        print(f"OTP for {to_email}: {otp}")
        print("="*50 + "\n\n")
        return

    message = Mail(
        from_email=from_email,
        to_emails=to_email,
        subject='Your OTP for DataVizAI',
        html_content=f'Your OTP is: <strong>{otp}</strong>')
    try:
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        print(f"OTP sent to {to_email}, status code: {response.status_code}")
    except Exception as e:
        print(f"Failed to send OTP email: {e}")
        print(f"OTP for {to_email}: {otp}")
