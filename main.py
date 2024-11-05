from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import io
import os
import json

# Initialize the FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)

# Load the model
model_path = os.path.join('model', 'BANICP.h5')
model = load_model(model_path)

# Define the class labels
CLASS_LABELS = {0: "Benign", 1: "Malignant"}

# In-memory storage for predictions history
history = []

def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess the image to prepare it for model prediction."""
    image = image.resize((128, 128))  # Resize to match model input
    image = img_to_array(image)         # Convert to array
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    image = image.astype("float32") / 255.0  # Normalize
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...), firstName: str = "", lastName: str = "", age: str = "", phoneNumber: str = "", keyword: str = ""):
    """Receive an image file, preprocess it, and predict using the CNN model."""
    try:
        # Load the uploaded image file
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Predict the class
        prediction = model.predict(processed_image)
        predicted_class = int(prediction[0][0] > 0.5)  # Threshold for binary classification
        class_label = CLASS_LABELS[predicted_class]

        # Store the prediction result in history
        history.append({
            "firstName": firstName,
            "lastName": lastName,
            "age": age,
            "phoneNumber": phoneNumber,
            "keyword": keyword,
            "label": class_label,
            "probability": float(prediction[0][0]),
        })

        # Return the result
        return {"label": class_label, "probability": float(prediction[0][0])}

    except Exception as e:
        return {"error": str(e)}

@app.get("/history/{keyword}")
async def get_history(keyword: str):
    """Retrieve the prediction history for a given keyword."""
    filtered_history = [entry for entry in history if entry["keyword"] == keyword]
    return {"history": filtered_history}

@app.get("/")
async def home():
    """Simple endpoint to verify the server is running."""
    return {"message": "BANICP MODEL API!"}
