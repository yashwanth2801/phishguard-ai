"""
benchmark.py — Compare LLaMA models on phishing detection using Groq API (FREE)
Usage: python benchmark.py
Set GROQ_API_KEY environment variable before running.
"""

import os, json, time, csv
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

TEST_EMAILS = [
    {"id": 1, "label": "phishing", "email": "Subject: URGENT: Verify Your Bank Account\nFrom: security@bankofamerica-alert.net\nYour account is flagged. Click: http://boa-verify.xyz/login\nProvide SSN and PIN to restore access."},
    {"id": 2, "label": "legitimate", "email": "Subject: Your monthly statement is ready\nFrom: statements@chase.com\nYour October statement for account ending 4521 is available at chase.com."},
    {"id": 3, "label": "phishing", "email": "Subject: You won $1,000,000!\nFrom: winner@lotto-international.org\nSend $200 processing fee via wire transfer to claim your prize."},
    {"id": 4, "label": "legitimate", "email": "Subject: Password changed successfully\nFrom: noreply@github.com\nYour GitHub password was changed. Contact support if you did not do this."},
    {"id": 5, "label": "phishing", "email": "Subject: IT Dept: Update credentials\nFrom: it-support@company-helpdesk.biz\nReply with your username and current password to update VPN access."},
]

SYSTEM_PROMPT = """Analyze the email. Respond ONLY with JSON:
{"classification": "phishing" or "legitimate", "confidence": <0-100>}"""

MODELS = ["llama3-70b-8192", "llama3-8b-8192"]


def predict(email_text, model):
    start = time.time()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": f"Email:\n{email_text}"}],
        temperature=0.1, max_tokens=100
    )
    elapsed = round(time.time() - start, 2)
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1].lstrip("json").strip()
    result = json.loads(raw)
    result["response_time"] = elapsed
    result["tokens"] = response.usage.total_tokens
    return result


def compute_metrics(results):
    tp = sum(1 for r in results if r["label"] == "phishing" and r["prediction"] == "phishing")
    tn = sum(1 for r in results if r["label"] == "legitimate" and r["prediction"] == "legitimate")
    fp = sum(1 for r in results if r["label"] == "legitimate" and r["prediction"] == "phishing")
    fn = sum(1 for r in results if r["label"] == "phishing" and r["prediction"] == "legitimate")
    total = len(results)
    accuracy  = round((tp + tn) / total * 100, 2) if total else 0
    precision = round(tp / (tp + fp) * 100, 2) if (tp + fp) else 0
    recall    = round(tp / (tp + fn) * 100, 2) if (tp + fn) else 0
    f1        = round(2 * precision * recall / (precision + recall), 2) if (precision + recall) else 0
    avg_time  = round(sum(r["response_time"] for r in results) / len(results), 2)
    avg_tok   = round(sum(r["tokens"] for r in results) / len(results))
    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "f1_score": f1, "avg_response_time": avg_time, "avg_tokens": avg_tok}


def run_benchmark():
    print("\n" + "="*60)
    print("  PhishGuard AI — Groq Benchmarking (FREE)")
    print("  CS 599 | Yashwanth Gundla & Tharun Javaji")
    print("="*60 + "\n")

    all_results = {m: [] for m in MODELS}

    for ed in TEST_EMAILS:
        print(f"Email #{ed['id']} (truth: {ed['label'].upper()})")
        for model in MODELS:
            try:
                pred = predict(ed["email"], model)
                correct = pred["classification"] == ed["label"]
                print(f"  [{model}] -> {pred['classification'].upper()} ({pred['confidence']}%) {'OK' if correct else 'WRONG'} [{pred['response_time']}s]")
                all_results[model].append({"id": ed["id"], "label": ed["label"],
                    "prediction": pred["classification"], "confidence": pred["confidence"],
                    "response_time": pred["response_time"], "tokens": pred["tokens"], "correct": correct})
            except Exception as e:
                print(f"  [{model}] ERROR: {e}")
        print()

    print("="*60)
    print("  RESULTS SUMMARY")
    print("="*60)
    metrics_all = {m: compute_metrics(all_results[m]) for m in MODELS}
    keys   = ["accuracy", "precision", "recall", "f1_score", "avg_response_time", "avg_tokens"]
    labels = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)", "Avg Time (s)", "Avg Tokens"]
    print(f"{'Metric':<25} {'LLaMA3-70B':>14} {'LLaMA3-8B':>12}")
    print("-"*55)
    for k, l in zip(keys, labels):
        vals = [str(metrics_all[m][k]) for m in MODELS]
        print(f"  {l:<23} {vals[0]:>14} {vals[1]:>12}")
    print("="*60)

    with open("benchmark_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","email_id","label","prediction","confidence","response_time","tokens","correct"])
        for m in MODELS:
            for r in all_results[m]:
                w.writerow([m, r["id"], r["label"], r["prediction"],
                            r["confidence"], r["response_time"], r["tokens"], r["correct"]])
    print("\n  Saved to benchmark_results.csv\n")


if __name__ == "__main__":
    run_benchmark()
