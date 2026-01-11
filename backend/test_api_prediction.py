import requests
import json

url = "http://localhost:8000/api/patients?patient_name=TestBot"
data = {
    "Age": 45,
    "Gender": "Male",
    "Admission_Type": "Emergency",
    "Department": "Cardiology",
    "Comorbidity": "None",
    "Procedures": 1
}

try:
    response = requests.post(url, json=data)
    print(f"Status Code: {response.status_code}")
    print("Response:", response.text)
except Exception as e:
    print(f"Request failed: {e}")
