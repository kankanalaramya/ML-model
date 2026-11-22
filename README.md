# ML-model
# Google Cloud Breast Cancer Prediction API – Progress Update

I have completed the initial setup and configuration work for our Breast Cancer Prediction API project using Google Cloud.

### 1. Machine Learning Model and API Configuration
- Verified that the trained Logistic Regression model loads successfully.
- Confirmed that the model expects 30 input features and has classes 0 and 1.
- Checked the prediction logic in `app.py` and corrected the class mapping so that 0 represents malignant and 1 represents benign.
- Verified that the application contains all 30 feature names and checked the Python syntax.

### 2. Cloud Storage and Secret Manager
- Verified that the Cloud Storage bucket `breast-cancer-model` is accessible.
- Confirmed that the trained model file `breast_cancer_model.pkl` exists in the bucket.
- Checked access to the Secret Manager secret `hospital-a-api-key`.
- Both Cloud Storage and Secret Manager access were successful using the active Google Cloud notebook service account.

### 3. Artifact Registry Setup
- Created a separate Docker repository named `ramya-prediction-api` in Artifact Registry.
- Configured it in the `northamerica-northeast2` region.
- Verified that the repository is accessible from our Google Cloud environment.

### 4. Docker Configuration
- Verified that Docker Engine is installed and running in Google Cloud JupyterLab.
- Checked the existing Dockerfile and `requirements.txt`.
- Confirmed that the Dockerfile uses Python 3.12, installs the required dependencies, copies the FastAPI application, exposes port 8080, and starts the application using Uvicorn.

### 5. Current Progress

The initial API configuration, cloud resource access checks, Artifact Registry setup, and Docker configuration are complete.

The next step is to build the Docker image and push it to our Artifact Registry repository.

After that, I will deploy the API to Cloud Run and test the prediction endpoint, API-key authentication, and application logging.

### Remaining Tasks
1. Build the Docker image.
2. Push the image to Artifact Registry.
3. Deploy the API to Cloud Run.
4. Test the health-check and prediction endpoints.
5. Verify API-key authentication.
6. Verify request logging.

**Overall status:** The preparation and configuration stages are complete. Docker image building and deployment are the next stages.
