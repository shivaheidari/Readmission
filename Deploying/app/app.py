from flask import request, jsonify, Flask, render_template
from model.predict import predict
from model.load_model import load_model
from transformers import AutoTokenizer

app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """
    Root endpoint to verify API is running.
    """
    return jsonify({"message": "Welcome to the Readmission Prediction API!!"})

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 

@app.route("/predict", methods=["POST", "GET"])
def predict_endpoint():
    model_path = "Readmission/Deploying/app/00_66_bert_custom_dict"
    model = load_model(model_path)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    try:
        if 'file' not in request.files:
            #return jsonify({"error": "No file provided"}), 400
            return render_template("index.html", prediction="No file uploaded.")
        
        file = request.files["file"]
        if not file:
           return jsonify({"error": "File is empty"}), 400

        #data = request.get_json()  # Get JSON input from Postman
        #text = data.get("text", "")  # Extract the text input
        text = file.read().decode("utf-8")
        # if not text:
        #     return jsonify({"error": "No text provided"}), 400  # Error handling
        
        results = predict(model, tokenizer, text)  # Call your prediction function
        
        #return jsonify({"label": results["predicted_class"]})  # Return response
        return render_template("index.html", prediction=f"Predicted Class: {results['predicted_class']}")

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
