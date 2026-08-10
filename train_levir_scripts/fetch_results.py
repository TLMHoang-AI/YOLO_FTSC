#!/usr/bin/env python3
import requests
import json
import base64

url = "https://sb-0c3b1c13bb2dc103.sb.molab.run/"
token = "8e76111a1f8cf917259eee057caf177e212a73c7ced6c11fe55fdfe15c28f139"

payload = {
    "code": """
import os
import base64
path = "/marimo/yolo_code/runs/eval_nms_05_results.json"
if os.path.exists(path):
    with open(path, "rb") as f:
        print(base64.b64encode(f.read()).decode("utf-8"))
else:
    print("NOT_FOUND")
"""
}

headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

print("Fetching results from remote server...")
response = requests.post(f"{url.rstrip('/')}/api/kernel/execute", headers=headers, json=payload)
if response.status_code == 200:
    res_data = response.json()
    outputs = res_data.get("outputs", [])
    b64_str = ""
    for out in outputs:
        if out.get("channel") == "stdout":
            b64_str += out.get("data", "")
            
    b64_str = b64_str.strip()
    if b64_str == "NOT_FOUND":
        print("Results file not found on remote server.")
    else:
        try:
            raw_bytes = base64.b64decode(b64_str)
            results = json.loads(raw_bytes)
            with open("./runs/eval_nms_05_results.json", "w") as f:
                json.dump(results, f, indent=2)
            print("Successfully saved runs/eval_nms_05_results.json locally!")
        except Exception as e:
            print("Error decoding base64 / parsing JSON:", e)
            print("Raw response:", b64_str[:500])
else:
    print("Failed to connect to remote kernel. Status code:", response.status_code)
