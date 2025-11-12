# 🧠 Titanic Survival Prediction API

A simple Flask API for predicting Titanic survival using a trained machine learning model.

## 📦 Project Structure

```
titanic_ml_api/
├── app.py                # Flask API
├── model.py              # Model training script
├── model.joblib          # Saved model (generated after running model.py)
├── preprocess.py         # Preprocessing logic (included in model.py)
├── test_request.py       # API client test
└── data/
    └── train.csv         # Titanic dataset
```

## 🚀 How to Use

1. Train the model:
```bash
python model.py
```

2. Run the API:
```bash
python app.py
```

3. Test the API:
```bash
python test_request.py
```

## 🔧 Install Dependencies

```bash
pip install -r requirements.txt
```

## 📜 License

MIT License
