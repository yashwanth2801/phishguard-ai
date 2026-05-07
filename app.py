from flask import Flask, render_template, request, jsonify
from groq import Groq
import os, json, time, threading

app = Flask(__name__)
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
scan_sessions = {}

SYSTEM_PROMPT = """You are a cybersecurity expert specializing in phishing email detection.
Respond ONLY with this exact JSON and nothing else:
{
  "classification": "phishing" or "legitimate",
  "confidence": <0-100>,
  "explanation": "<detailed explanation>",
  "red_flags": ["<flag1>", "<flag2>"],
  "recommendation": "<action the user should take>"
}
red_flags should be [] if legitimate. Do not include any text outside the JSON."""

def analyze_single(email_text, model="llama3-70b-8192"):
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze this email:\n\n{email_text}"}
        ],
        temperature=0.2,
        max_tokens=800
    )
    elapsed = round(time.time() - start, 2)
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    result = json.loads(raw)
    result["model"] = model
    result["response_time"] = elapsed
    result["tokens_used"] = response.usage.total_tokens
    return result

@app.route("/")
def index(): return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("email_text", "").strip()
    model = data.get("model", "llama3-70b-8192")
    if not text: return jsonify({"error": "No email text provided"}), 400
    try: return jsonify(analyze_single(text, model))
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/scan/start", methods=["POST"])
def scan_start():
    from email_scanner import scan_inbox, detect_provider
    data = request.get_json()
    email_addr = data.get("email")
    password = data.get("password")
    limit = int(data.get("limit", 20))
    model = data.get("model", "llama3-70b-8192")
    custom_host = data.get("custom_host", None)
    if not email_addr or not password:
        return jsonify({"error": "Email and password required"}), 400
    sid = str(int(time.time() * 1000))
    scan_sessions[sid] = {"status": "running", "progress": 0, "total": limit,
                          "results": [], "summary": None, "error": None,
                          "provider": detect_provider(email_addr)}
    def run():
        def cb(current, total, result):
            scan_sessions[sid]["progress"] = current
            scan_sessions[sid]["total"] = total
            scan_sessions[sid]["results"].append({
                "id": result.get("id"), "subject": result.get("subject"),
                "sender": result.get("sender"), "date": result.get("date"),
                "preview": result.get("preview", ""),
                "classification": result.get("classification"),
                "confidence": result.get("confidence", 0),
                "explanation": result.get("explanation", ""),
                "red_flags": result.get("red_flags", []),
                "error": result.get("error")
            })
        try:
            s = scan_inbox(email_addr, password, limit, model, custom_host, cb)
            scan_sessions[sid]["summary"] = {"total": s["total"], "phishing": s["phishing"], "legitimate": s["legitimate"]}
            scan_sessions[sid]["status"] = "done"
        except Exception as e:
            scan_sessions[sid]["status"] = "error"
            scan_sessions[sid]["error"] = str(e)
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"session_id": sid})

@app.route("/scan/status/<sid>")
def scan_status(sid):
    s = scan_sessions.get(sid)
    if not s: return jsonify({"error": "Session not found"}), 404
    return jsonify(s)

@app.route("/benchmark", methods=["POST"])
def benchmark():
    data = request.get_json()
    text = data.get("email_text", "").strip()
    models = data.get("models", ["llama3-70b-8192", "llama3-8b-8192"])
    if not text: return jsonify({"error": "No email provided"}), 400
    results = {}
    for m in models:
        try: results[m] = analyze_single(text, m)
        except Exception as e: results[m] = {"error": str(e)}
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
