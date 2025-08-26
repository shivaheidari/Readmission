import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

API_KEY = os.getenv("API_KEY")

if API_KEY:

    genai.configure(api_key=API_KEY)
else:
    print("Warning: GEMNI_API_KEY not found")

