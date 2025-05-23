import requests
import json

def test_intent_classification():
    url = "http://localhost:8000/api/classify-intent"
    payload = {
        "message": "I'm looking for something cheaper",
        "min_similarity_threshold": 0.7,
        "limit": 1
    }
    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, json=payload, headers=headers)
        print(f"Status Code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    test_intent_classification() 