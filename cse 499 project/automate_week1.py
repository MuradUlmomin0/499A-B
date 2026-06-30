import os
import subprocess
import sys

def main():
    # Step 1: Programmatically install 'fastapi' and 'uvicorn' via pip
    print("Installing fastapi and uvicorn...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fastapi", "uvicorn"])
        print("fastapi and uvicorn installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

    # Step 2: Create a 'main.py' file containing a GET endpoint at '/ping' that returns {"status": "ok"}
    main_py_content = '''from fastapi import FastAPI
from fastapi.responses import HTMLResponse

app = FastAPI()

html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PRIVATE SHIELD // Federated Intelligence Engine</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;500;600&display=swap');

    :root {
      --bg-color: #060b19;
      --card-bg: rgba(10, 25, 47, 0.4);
      --accent-teal: #00f5d4;
      --accent-cyan: #00bbf9;
      --border-glow: rgba(0, 245, 212, 0.15);
      --text-main: #f8fafc;
      --text-muted: #94a3b8;
    }

    body {
      margin: 0;
      padding: 0;
      background-color: var(--bg-color);
      background-image: 
        radial-gradient(circle at 50% 50%, rgba(0, 187, 249, 0.12) 0%, transparent 60%);
      font-family: 'Inter', sans-serif;
      color: var(--text-main);
      min-height: 100vh;
      display: flex;
      justify-content: center;
      align-items: center;
      box-sizing: border-box;
      padding: 20px;
    }

    .container {
      width: 100%;
      max-width: 580px;
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      -webkit-backdrop-filter: blur(16px);
      border: 1px solid var(--border-glow);
      border-radius: 20px;
      padding: 40px;
      box-sizing: border-box;
      box-shadow: 
        0 30px 60px rgba(0, 0, 0, 0.6),
        0 0 40px rgba(0, 245, 212, 0.03);
      animation: loadIn 0.8s cubic-bezier(0.16, 1, 0.3, 1);
      text-align: center;
    }

    @keyframes loadIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .header-group {
      margin-bottom: 35px;
    }

    .system-title {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.4rem;
      font-weight: 700;
      letter-spacing: 2px;
      background: linear-gradient(90deg, var(--accent-cyan), var(--accent-teal));
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin: 0 0 8px 0;
      text-shadow: 0 0 30px rgba(0, 187, 249, 0.2);
    }

    .system-subtitle {
      font-size: 0.85rem;
      color: var(--text-muted);
      letter-spacing: 1px;
      text-transform: uppercase;
      margin: 0 0 20px 0;
    }

    .status-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      font-family: 'Orbitron', sans-serif;
      font-size: 0.75rem;
      font-weight: 600;
      color: var(--accent-teal);
      background: rgba(0, 245, 212, 0.08);
      padding: 6px 14px;
      border-radius: 50px;
      border: 1px solid rgba(0, 245, 212, 0.25);
      text-transform: uppercase;
      letter-spacing: 1px;
    }

    .pulse-dot {
      width: 6px;
      height: 6px;
      background-color: var(--accent-teal);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--accent-teal);
      animation: pulse 2s infinite;
    }

    @keyframes pulse {
      0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 245, 212, 0.5); }
      70% { transform: scale(1); box-shadow: 0 0 0 6px rgba(0, 245, 212, 0); }
      100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 245, 212, 0); }
    }

    .divider {
      height: 1px;
      background: linear-gradient(90deg, transparent, var(--border-glow), transparent);
      margin: 30px 0;
    }

    .metrics-container {
      display: grid;
      grid-template-columns: 1fr 1fr 1fr;
      gap: 15px;
    }

    @media (max-width: 480px) {
      .metrics-container {
        grid-template-columns: 1fr;
        gap: 20px;
      }
    }

    .metric-card {
      padding: 15px 10px;
      border-radius: 12px;
      background: rgba(255, 255, 255, 0.01);
      border: 1px solid rgba(255, 255, 255, 0.03);
      transition: all 0.3s ease;
    }

    .metric-card:hover {
      background: rgba(255, 255, 255, 0.03);
      border-color: rgba(0, 187, 249, 0.2);
      transform: translateY(-2px);
    }

    .metric-label {
      font-size: 0.75rem;
      color: var(--text-muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      display: block;
      margin-bottom: 8px;
    }

    .metric-value {
      font-family: 'Orbitron', sans-serif;
      font-size: 1.15rem;
      font-weight: 700;
    }

    .val-online {
      color: var(--accent-teal);
      text-shadow: 0 0 10px rgba(0, 245, 212, 0.3);
    }

    .val-nodes {
      color: var(--accent-cyan);
      text-shadow: 0 0 10px rgba(0, 187, 249, 0.3);
    }

    .val-threats {
      color: #94a3b8;
    }
  </style>
</head>
<body>

<div class="container">
  <div class="header-group">
    <h1 class="system-title">PRIVATE SHIELD</h1>
    <h2 class="system-subtitle">Federated Intelligence Engine</h2>
    <div class="status-badge">
      <div class="pulse-dot"></div>
      Engine Active
    </div>
  </div>

  <div class="divider"></div>

  <div class="metrics-container">
    <div class="metric-card">
      <span class="metric-label">Network State</span>
      <span class="metric-value val-online">ONLINE</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Connected Nodes</span>
      <span class="metric-value val-nodes">3</span>
    </div>
    <div class="metric-card">
      <span class="metric-label">Active Threats</span>
      <span class="metric-value val-threats">0</span>
    </div>
  </div>
</div>

</body>
</html>
"""

@app.get("/ping")
def ping(format: str = None):
    if format == "json":
        return {"status": "ok"}
    return HTMLResponse(content=html_content)
'''
    print("Creating main.py...")
    try:
        with open("main.py", "w") as f:
            f.write(main_py_content)
        print("main.py created successfully!")
    except Exception as e:
        print(f"Error creating main.py: {e}")
        sys.exit(1)

    # Step 3: Automatically launch the Uvicorn server on localhost port 8000
    print("Launching Uvicorn server on localhost:8000...")
    try:
        # We run uvicorn as a subprocess of the current python interpreter
        # using the command: python -m uvicorn main:app --host 127.0.0.1 --port 8000
        # This is cross-platform and reliable.
        subprocess.run([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"])
    except Exception as e:
        print(f"Error launching Uvicorn server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
