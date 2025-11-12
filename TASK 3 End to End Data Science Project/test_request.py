import requests

url = 'http://127.0.0.1:5000/predict'
data = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22,
    "SibSp": 1,
    "Fare": 7.25
}

response = requests.post(url, json=data)
print(response.json())
