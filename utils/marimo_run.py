#!/usr/bin/env python3
import sys
import os
import json
import requests
import argparse

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "marimo_config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {"url": "", "token": ""}

def save_config(url, token):
    with open(CONFIG_PATH, "w") as f:
        json.dump({"url": url, "token": token}, f, indent=2)
    print(f"Saved configuration to {CONFIG_PATH}")

def run_code_on_server(code, config):
    url = config.get("url", "").rstrip("/")
    token = config.get("token", "")
    if not url or not token:
        print("Error: Marimo server URL and Token are not configured.")
        print("Please configure them using: python3 utils/marimo_run.py --set-config URL TOKEN")
        sys.exit(1)
        
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    # 1. Discover session ID
    try:
        r = requests.get(f"{url}/api/sessions", headers=headers)
        if r.status_code != 200:
            print(f"Failed to connect to Marimo server. Status: {r.status_code}")
            sys.exit(1)
        sessions = r.json()
        if not sessions:
            print("No active Marimo sessions found. Open the notebook in your browser.")
            sys.exit(1)
        session_id = list(sessions.keys())[0]
    except Exception as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    # 2. Execute code via execute endpoint
    headers["Marimo-Session-Id"] = session_id
    payload = {"code": code}
    
    try:
        # Use SSE stream mode to print output in real-time
        r = requests.post(f"{url}/api/kernel/execute", headers=headers, json=payload, stream=True)
        if r.status_code != 200:
            print(f"Failed to execute code. Status: {r.status_code}")
            sys.exit(1)
            
        current_event = ""
        for line in r.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8").strip()
            if line.startswith("event:"):
                current_event = line.replace("event:", "").strip()
            elif line.startswith("data:"):
                data_str = line.replace("data:", "").strip()
                try:
                    payload_data = json.loads(data_str)
                    if current_event == "stdout":
                        sys.stdout.write(payload_data.get("data", ""))
                        sys.stdout.flush()
                    elif current_event == "stderr":
                        sys.stderr.write(payload_data.get("data", ""))
                        sys.stderr.flush()
                    elif current_event == "done":
                        if payload_data.get("success") is False:
                            print(f"\nExecution error: {payload_data.get('error', {}).get('msg', 'Unknown error')}", file=sys.stderr)
                            sys.exit(1)
                        else:
                            val = payload_data.get("output", {}).get("data", "")
                            if val:
                                print(f"\nResult: {val}")
                        break
                except Exception:
                    pass
    except Exception as e:
        print(f"Error during execution: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Marimo remote helper utility")
    parser.add_argument("--set-config", nargs=2, metavar=("URL", "TOKEN"), help="Set remote Marimo server URL and token")
    parser.add_argument("-c", "--code", help="Run Python code snippet on remote server")
    parser.add_argument("-f", "--file", help="Run a local Python file on remote server")
    
    args = parser.parse_args()
    config = load_config()
    
    if args.set_config:
        save_config(args.set_config[0], args.set_config[1])
        sys.exit(0)
        
    if args.code:
        run_code_on_server(args.code, config)
    elif args.file:
        if not os.path.exists(args.file):
            print(f"Local file not found: {args.file}")
            sys.exit(1)
        with open(args.file, "r") as f:
            code = f.read()
        run_code_on_server(code, config)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
