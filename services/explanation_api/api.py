import os
import google.generativeai as genai

api_key = os.environ.get("API_KEY")

print(api_key)

# model = genai.GenerationConfig("gemini-1.5-pro")

# response = model.genrate_content("Explain docker in simple terms")

# print(response.text)