import requests


# Send a GET request and print the greeting.
r = requests.get("http://127.0.0.1:8000/", timeout=30)
print("GET status code:", r.status_code)
print("Welcome message:", r.json())
r.raise_for_status()

data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST request and print the prediction.
r = requests.post(
    "http://127.0.0.1:8000/data/",
    json=data,
    timeout=30,
)
print("POST status code:", r.status_code)
print("Prediction result:", r.json())
r.raise_for_status()
