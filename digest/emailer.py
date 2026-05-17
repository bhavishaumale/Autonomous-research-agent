# digest/emailer.py

import smtplib
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def send_email_digest(digest_markdown: str):
    """Sends the weekly digest via Gmail SMTP."""

    sender = os.getenv("EMAIL_FROM")
    recipient = os.getenv("EMAIL_TO")
    app_password = os.getenv("GMAIL_APP_PASSWORD")

    if not all([sender, recipient, app_password]):
        raise ValueError(
            "Missing email credentials. Check EMAIL_FROM, EMAIL_TO, "
            "and GMAIL_APP_PASSWORD in your .env file."
        )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Your Weekly AI Research Digest"
    msg["From"] = sender
    msg["To"] = recipient

    plain_part = MIMEText(digest_markdown, "plain", "utf-8")
    msg.attach(plain_part)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender, app_password)
        server.sendmail(sender, recipient, msg.as_string())
        print("Email sent successfully!")