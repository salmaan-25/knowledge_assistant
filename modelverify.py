# Run this once to check available models:
import os

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
for m in genai.list_models():
    print(m.name, m.supported_generation_methods)
