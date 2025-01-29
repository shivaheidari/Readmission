from flask import request, jsonify, Flask
from model.predict import predict



app = Flask(__name__)


@app.route("/", methods=["GET"])
def home():
    """
    Root endpoint to verify API is running.
    """
    return jsonify({"message": "API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint to make predictions using the model.
    """
    try:
        # Parse input JSON
        data = request.json
        if "text" not in data:
            return jsonify({"error": "Missing 'text' field in the request body"}), 400

        input_text = data["text"]

        # Call the predict function from predict.py
        prediction_result = predict(input_text)

        # Return the prediction result
        return jsonify(prediction_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
