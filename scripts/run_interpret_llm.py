from config import API_KEY
print(API_KEY)



# # In your api.py, add these imports at the top
# from openai import OpenAI
# import os

# # --- Add this after you create the SHAP explainer ---
# # Initialize the OpenAI client
# try:
#     client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
# except TypeError:
#     client = None
#     print("OpenAI API key not found. Narrative generation will be disabled.")


# # --- Replace your existing /v1/explain-readmission endpoint with this ---

# @app.post("/v1/explain-readmission")
# def explain_readmission(note: InputNote):
#     """
#     Accepts a clinical note and returns a SHAP-based explanation
#     with a human-readable narrative summary.
#     """
#     cleaned_text = clean_mimic_text(note.text)
#     shap_values = explainer([cleaned_text])
    
#     # --- Synthesize SHAP data ---
#     positive_class_explanation = shap_values[0, :, "LABEL_1"]
#     words = positive_class_explanation.data
#     impacts = positive_class_explanation.values
    
#     explanation_data = [{"word": str(word), "impact": round(float(impact), 4)} for word, impact in zip(words, impacts) if word is not None]
    
#     # Get top 5 positive and top 3 negative keywords
#     explanation_data.sort(key=lambda x: x['impact'], reverse=True)
#     top_positive_words = [item['word'] for item in explanation_data[:5] if item['impact'] > 0]
#     top_negative_words = [item['word'] for item in explanation_data[-3:] if item['impact'] < 0]

#     # --- Generate Narrative with LLM ---
#     narrative_summary = "Narrative generation disabled. OpenAI client not available."
#     if client and top_positive_words:
#         prompt = (
#             "You are a clinical AI assistant. A machine learning model predicted a patient's 30-day readmission risk. "
#             "Explain the prediction to a clinician in 2-3 concise sentences. Be direct.\n\n"
#             f"The model's prediction is 'High Risk'.\n"
#             f"The primary factors INCREASING this risk were mentions of: {', '.join(top_positive_words)}.\n"
#             f"The primary factors DECREASING this risk were mentions of: {', '.join(top_negative_words) if top_negative_words else 'none significant'}.\n\n"
#             "Summary:"
#         )
        
#         response = client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.5,
#             max_tokens=150
#         )
#         narrative_summary = response.choices[0].message.content

#     # --- Return Enriched Response ---
#     return {
#         "hadm_id": note.hadm_id,
#         "narrative_summary": narrative_summary,
#         "explanation_data": explanation_data
#     }