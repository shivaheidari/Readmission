from fastapi.testclient import TestClient
from api import app


client = TestClient(app)
def test_read_main():
    sample_note = {
      "hadm_id": 12345,
      "text": "The patient reports shortness of breath. History of congestive heart failure."
    }

    response = client.post("/v1/predict-readmission", json=sample_note)
    assert response.status_code == 200

    response_json = response.json()
    assert response_json["resourceType"] == "RiskAssessment"
    assert "prediction" in response_json
    assert response_json["prediction"][0]["outcome"]["text"] in ["High Risk", "Low Risk"]

def test_predict_readmission_invalid_input():
    """Tests that the API handles invalid input gracefully."""
    # Send a request with a missing 'text' field
    invalid_note = {"hadm_id": 54321}
    
    response = client.post("/v1/predict-readmission", json=invalid_note)
    
    # FastAPI should return a 422 Unprocessable Entity error for validation failures
    assert response.status_code == 422

    