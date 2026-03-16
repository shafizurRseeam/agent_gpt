import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "phi3",
    "prompt": "Break this task into steps: book dentist appointment",
    "stream": False
}

response = requests.post(url, json=payload)

print(response.json()["response"])