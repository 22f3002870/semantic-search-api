import urllib.request
import json
import time

url = "http://localhost:8000/search"
payload = {
    "query": "battery life issues",
    "k": 5,
    "rerank": True,
    "rerankK": 3
}

print(f"Testing {url}...")
try:
    start = time.time()
    req = urllib.request.Request(
        url, 
        data=json.dumps(payload).encode('utf-8'), 
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req) as response:
        end = time.time()
        if response.status == 200:
            data = json.load(response)
            print("Success!")
            print(json.dumps(data, indent=2))
            print(f"Request took {end - start:.2f}s")
        else:
            print(f"Failed with status {response.status}")
            print(response.read().decode('utf-8'))

except Exception as e:
    print(f"Error: {e}")
