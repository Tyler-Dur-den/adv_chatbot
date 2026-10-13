import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("GOOGLE_API_KEY not found in .env")
else:
    url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
    response = requests.get(url)
    
    if response.status_code == 200:
        models = response.json().get('models', [])
        print("Available Google Models:")
        for m in models:
            # Just printing the base names so it's easy to read
            print(f"- {m['name'].replace('models/', '')}")
    else:
        print(f"Error: {response.status_code} - {response.text}")