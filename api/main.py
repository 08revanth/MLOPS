from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import cv2
import io
from PIL import Image

app = FastAPI(title="Diabetic Retinopathy Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows any website/local file to access the API
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
MODEL = tf.keras.models.load_model('models/final_model.keras')
CLASSES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

@app.get("/")
def home():
    return {"message": "Diabetic Retinopathy API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # 1. Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('RGB')
    
    # 2. Preprocess (Resize and Normalize)
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    
    # 3. Predict
    predictions = MODEL.predict(img_array)
    class_idx = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0]))
    
    return {
        "filename": file.filename,
        "prediction": CLASSES[class_idx],
        "severity_level": int(class_idx),
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)