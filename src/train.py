import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import build_model
import mlflow
import mlflow.keras
import os

# 1. Setup absolute paths to avoid confusion
BASE_DIR = os.path.abspath(os.path.join(os.getcwd()))
IMG_DIR = os.path.join(BASE_DIR, "data", "raw")
CSV_PATH = os.path.join(IMG_DIR, "train.csv")

# 2. Load Data
df = pd.read_csv(CSV_PATH)
df['id_code'] = df['id_code'].apply(lambda x: f"{x}.png")
df['diagnosis'] = df['diagnosis'].astype(str)

# 3. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2
)

# 4. Generators
train_generator = train_datagen.flow_from_dataframe(
    df, 
    directory=IMG_DIR, 
    x_col='id_code', 
    y_col='diagnosis',
    target_size=(224, 224), 
    batch_size=16, # Lower batch size to prevent memory errors on CPU
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_dataframe(
    df, 
    directory=IMG_DIR, 
    x_col='id_code', 
    y_col='diagnosis',
    target_size=(224, 224), 
    batch_size=16,
    class_mode='categorical',
    subset='validation'
)

# 5. Training
mlflow.set_experiment("Diabetic_Retinopathy")
with mlflow.start_run():
    # Use the weights we calculated
    class_weights = {0: 0.40, 1: 1.97, 2: 0.73, 3: 3.79, 4: 2.48}
    
    model = build_model()
    
    # Fast training for hackathon: 5 epochs, few steps per epoch
    model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=5,
        steps_per_epoch=20, # Remove this for full training
        class_weight=class_weights
    )
    
    if not os.path.exists("models"): os.makedirs("models")
    model.save('models/final_model.keras')
    print("Model Saved!")