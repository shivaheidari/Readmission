from flask import request, jsonify, Flask
from model.predict import predict



app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """
    Root endpoint to verify API is running.
    """
    return jsonify({"message": "Welcome to the Readmission Prediction API!!"})

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json  # Expecting JSON input
    text = data.get("text")

    if not text:
        return jsonify({"error": "No text provided"}), 400  # Return error if text is missing

    # Call the prediction function
    prediction = predict(text)
    return jsonify({"prediction": prediction})  # Return prediction result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
