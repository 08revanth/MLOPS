import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model

def build_model(num_classes=5, learning_rate=1e-4):
    # 1. Load the backbone (Pre-trained on ImageNet)
    base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # Freeze the base model (Transfer Learning)
    base_model.trainable = False 
    
    # 2. Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # 3. Output layer (Softmax for multi-class)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    # 4. Assemble
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # 5. Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

if __name__ == "__main__":
    m = build_model()
    m.summary() # This verifies the architecture