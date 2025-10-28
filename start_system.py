import subprocess, time, os, sys
base = os.path.dirname(os.path.abspath(__file__))

print("🚀 Starting Detection System...")

subprocess.Popen(["python3", os.path.join(base, "src/app.py")])
time.sleep(3)
subprocess.Popen(["uvicorn", "feedback_api_proxy:app", "--port", "8000"], cwd=base)
time.sleep(3)
subprocess.Popen(["ngrok", "http", "8000"], cwd=base)

input("✅ System running. Press Enter to close.\n")