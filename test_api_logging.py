
import json
import zipfile
import pandas as pd

from fastapi.testclient import TestClient
from app import app

# Initialize FastAPI test client
client = TestClient(app)

# Load sample breast cancer features
zip_path = "breast+cancer+wisconsin+diagnostic.zip"

with zipfile.ZipFile(zip_path, "r") as z:
    with z.open("wdbc.data") as file:
        df = pd.read_csv(file, header=None)

sample_features = df.iloc[0, 2:].astype(float).tolist()

# Enter your Secret Manager API key locally
api_key = input("Enter your API key: ")

# Send prediction request
response = client.post(
    "/predict",
    headers={"x-api-key": api_key},
    json={"features": sample_features}
)

print("Status:", response.status_code)
print("Prediction response:")
print(json.dumps(response.json(), indent=4))
