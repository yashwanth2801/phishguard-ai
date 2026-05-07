import imaplib
import email
from email.header import decode_header
import json
import time
import os
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

IMAP_SERVERS = {
    "gmail":   {"host": "imap.gmail.com",        "port": 993},
    "outlook": {"host": "imap-mail.outlook.com",  "port": 993},
    "hotmail": {"host": "imap-mail.outlook.com",  "port": 993},
    "yahoo":   {"host": "imap.mail.yahoo.com",    "port": 993},
    "other":   {"host": None,                     "port": 993},
}

SYSTEM_PROMPT = """You are a cybersecurity expert. Analyze the email and respond ONLY with this JSON:
{
  "classification": "phishing" or "legitimate",
  "confidence": <0-100>,
  "explanation": "<brief explanation>",
  "red_flags": ["<flag1>", "<flag2>"]
}
red_flags should be [] if legitimate. No text outside the JSON."""


def detect_provider(email_address: str) -> str:
    domain = email_address.split("@")[-1].lower()
    if "gmail" in domain: return "gmail"
    elif any(x in domain for x in ["outlook", "hotmail", "live", "msn"]): return "outlook"
    elif "yahoo" in domain: return "yahoo"
    return "other"


def connect_imap(email_address, password, custom_host=None):
    provider = detect_provider(email_address)
    config = IMAP_SERVERS[provider]
    host = custom_host if custom_host else config["host"]
    if not host:
        raise ValueError("Unknown email provider. Please provide IMAP host.")
    mail = imaplib.IMAP4_SSL(host, config["port"])
    mail.login(email_address, password)
    return mail


def decode_str(s):
    if s is None: return ""
    decoded, enc = decode_header(s)[0]
    if isinstance(decoded, bytes):
        return decoded.decode(enc or "utf-8", errors="replace")
    return decoded


def get_email_body(msg):
    body = ""
    if msg.is_multipart():
        for part in msg.walk():
            ct = part.get_content_type()
            cd = str(part.get("Content-Disposition", ""))
            if ct == "text/plain" and "attachment" not in cd:
                try:
                    body = part.get_payload(decode=True).decode(
                        part.get_content_charset() or "utf-8", errors="replace")
                    break
                except: pass
    else:
        try:
            body = msg.get_payload(decode=True).decode(
                msg.get_content_charset() or "utf-8", errors="replace")
        except: pass
    return body[:3000]


def fetch_emails(email_address, password, limit=20, folder="INBOX", custom_host=None):
    mail = connect_imap(email_address, password, custom_host)
    mail.select(folder)
    _, data = mail.search(None, "ALL")
    email_ids = data[0].split()
    recent_ids = list(reversed(email_ids[-limit:] if len(email_ids) >= limit else email_ids))
    emails = []
    for eid in recent_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        raw = msg_data[0][1]
        msg = email.message_from_bytes(raw)
        subject = decode_str(msg.get("Subject", "(No Subject)"))
        sender = decode_str(msg.get("From", "Unknown"))
        date = msg.get("Date", "")
        body = get_email_body(msg)
        emails.append({"id": eid.decode(), "subject": subject, "sender": sender,
                       "date": date, "body": body,
                       "preview": body[:150].replace("\n", " ").strip()})
    mail.logout()
    return emails


def analyze_email_llm(email_data, model="llama3-70b-8192"):
    content = f"""Subject: {email_data['subject']}
From: {email_data['sender']}
Date: {email_data['date']}

{email_data['body']}"""
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this email:\n\n{content}"}
        ],
        temperature=0.2,
        max_tokens=500
    )
    elapsed = round(time.time() - start, 2)
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    result = json.loads(raw)
    result["response_time"] = elapsed
    result["tokens"] = response.usage.total_tokens
    return result


def scan_inbox(email_address, password, limit=20, model="llama3-70b-8192",
               custom_host=None, progress_callback=None):
    emails = fetch_emails(email_address, password, limit, custom_host=custom_host)
    results = []
    phishing_count = 0
    legitimate_count = 0
    for i, em in enumerate(emails):
        try:
            analysis = analyze_email_llm(em, model)
            record = {**em, **analysis}
            if analysis["classification"] == "phishing": phishing_count += 1
            else: legitimate_count += 1
            results.append(record)
        except Exception as e:
            results.append({**em, "classification": "error", "error": str(e),
                             "confidence": 0, "explanation": str(e), "red_flags": []})
        if progress_callback:
            progress_callback(i + 1, len(emails), results[-1])
    return {"total": len(results), "phishing": phishing_count,
            "legitimate": legitimate_count, "emails": results}
