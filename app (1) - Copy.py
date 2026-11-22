
import joblib
import json
import hashlib
import logging
import time

from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from google.cloud import storage
from google.cloud import secretmanager


# Google Cloud configuration
PROJECT_ID = "cloud-computing-project-510701"
BUCKET_NAME = "breast-cancer-model"
MODEL_FILE = "breast_cancer_model.pkl"
SECRET_ID = "hospital-a-api-key"


# Feature names in the exact order used by the dataset
FEATURE_NAMES = [
    "mean_radius",
    "mean_texture",
    "mean_perimeter",
    "mean_area",
    "mean_smoothness",
    "mean_compactness",
    "mean_concavity",
    "mean_concave_points",
    "mean_symmetry",
    "mean_fractal_dimension",
    "radius_error",
    "texture_error",
    "perimeter_error",
    "area_error",
    "smoothness_error",
    "compactness_error",
    "concavity_error",
    "concave_points_error",
    "symmetry_error",
    "fractal_dimension_error",
    "worst_radius",
    "worst_texture",
    "worst_perimeter",
    "worst_area",
    "worst_smoothness",
    "worst_compactness",
    "worst_concavity",
    "worst_concave_points",
    "worst_symmetry",
    "worst_fractal_dimension"
]


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prediction-api")


# Download model from Cloud Storage
storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blob = bucket.blob(MODEL_FILE)

blob.download_to_filename(MODEL_FILE)

# Load trained model
model = joblib.load(MODEL_FILE)

# Initialize Secret Manager
secret_client = secretmanager.SecretManagerServiceClient()


# Initialize FastAPI
app = FastAPI(
    title="Breast Cancer Prediction API",
    description="API for breast cancer prediction using Logistic Regression",
    version="1.0"
)

print("Model loaded successfully!")


# Request format
class PredictionRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=30,
        max_length=30,
        description="Exactly 30 breast cancer measurement features"
    )


# Verify API key using Secret Manager
def verify_api_key(x_api_key: str):

    secret_name = (
        f"projects/{PROJECT_ID}/secrets/"
        f"{SECRET_ID}/versions/latest"
    )

    try:
        response = secret_client.access_secret_version(
            request={"name": secret_name}
        )

        valid_api_key = response.payload.data.decode("UTF-8")

    except Exception:
        logger.exception("Secret Manager access failed")
        raise HTTPException(
            status_code=500,
            detail="Unable to verify API key"
        )

    if x_api_key != valid_api_key:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )


# Generate a stable pseudonymous caller ID
def generate_caller_id(api_key: str):
    key_hash = hashlib.sha256(
        api_key.encode("utf-8")
    ).hexdigest()

    return "key_" + key_hash[:12]


# Health check endpoint
@app.get("/")
def home():
    return {
        "message": "Breast Cancer Prediction API is running",
        "status": "healthy"
    }


# Prediction endpoint
@app.post("/predict")
def predict(
    payload: PredictionRequest,
    http_request: Request,
    x_api_key: str = Header(...)
):

    start_time = time.perf_counter()

    # Authenticate caller
    verify_api_key(x_api_key)

    try:
        # Make prediction
        prediction = model.predict([payload.features])[0]

        probabilities = model.predict_proba(
            [payload.features]
        )[0]

        result = (
        "malignant" if prediction == 0 else "benign"
        )

        confidence = float(max(probabilities))

        # Calculate request latency
        latency_ms = (
            time.perf_counter() - start_time
        ) * 1000

        # Get source IP
        source_ip = (
            http_request.client.host
            if http_request.client
            else "unknown"
        )

        # Create structured JSON log
        log_record = {
            "caller_id": generate_caller_id(x_api_key),
            "request_time": datetime.now(
                timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S.%f"),
            "source_ip": source_ip,

            **dict(zip(FEATURE_NAMES, payload.features)),

            "prediction": result,
            "confidence": round(confidence, 4),
            "latency_ms": round(latency_ms, 2)
        }

        # Write one JSON log line
        logger.info(
            json.dumps(log_record, separators=(",", ":"))
        )

        # Return prediction response
        return {
            "prediction": result,
            "confidence": round(confidence, 4)
        }

    except HTTPException:
        raise

    except Exception:
        logger.exception("Prediction request failed")

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )
