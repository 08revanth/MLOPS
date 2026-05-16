import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate_model():
    # 1. Load the trained model
    model = tf.keras.models.load_model('models/final_model.keras')
    
    # 2. Setup Validation Generator (No augmentation for evaluation)
    df = pd.read_csv("data/raw/train.csv")
    df['id_code'] = df['id_code'].apply(lambda x: f"{x}.png")
    df['diagnosis'] = df['diagnosis'].astype(str)
    
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    val_generator = val_datagen.flow_from_dataframe(
        df, directory="data/raw/", x_col='id_code', y_col='diagnosis',
        target_size=(224, 224), batch_size=16, class_mode='categorical',
        subset='validation', shuffle=False
    )
    
    # 3. Get Predictions
    print("Generating predictions...")
    y_pred_probs = model.predict(val_generator)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = val_generator.classes
    
    # 4. Calculate Quadratic Weighted Kappa (Gold standard for DR)
    kappa = cohen_kappa_score(y_true, y_pred, weights='quadratic')
    print(f"\nQuadratic Weighted Kappa Score: {kappa:.4f}")
    
    # 5. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No DR', 'Mild', 'Mod', 'Sev', 'Prolif'],
                yticklabels=['No DR', 'Mild', 'Mod', 'Sev', 'Prolif'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("[Saved] confusion_matrix.png")
    
    # 6. Detailed Report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

if __name__ == "__main__":
    evaluate_model()