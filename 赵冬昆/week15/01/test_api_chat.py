
"""Test API chat endpoint"""
import requests

url = "http://localhost:8000/api/chat"
headers = {"Content-Type": "application/json"}
data = {"query": "汽车发动机是什么？"}

try:
    response = requests.post(url, headers=headers, json=data)
    response.encoding = 'utf-8'
    print(f"Status code: {response.status_code}", flush=True)
    
    # Write to file to avoid console encoding issues
    with open("api_response.json", "w", encoding="utf-8") as f:
        f.write(response.text)
    print("Response written to api_response.json", flush=True)
    
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
