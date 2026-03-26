import os, sys
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("API_URL")

if not api_key:
    print("Error: API_URL not found in environment/.env")
    sys.exit(1)

# Mask the API key for safety
masked = api_key[:4] + "*" * (len(api_key)-8) + api_key[-4:]
print(f"Checking connectivity with API key: {masked}")

try:
    genai.configure(api_key=api_key)
    print("Listing available models that support generateContent:")
    models = genai.list_models()
    found = False
    for m in models:
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            found = True
    if not found:
        print("No models found with 'generateContent' support.")
except Exception as e:
    print(f"Critical Error: {e}")
