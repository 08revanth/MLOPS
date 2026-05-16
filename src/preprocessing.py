import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def run_eda():
    csv_path = "data/raw/train.csv"
    img_dir = "data/raw/"
    
    # 1. Load data
    df = pd.read_csv(csv_path)
    
    # 2. Map labels
    diagnosis_lookup = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative"
    }
    df['diagnosis_name'] = df['diagnosis'].map(diagnosis_lookup)
    
    # 3. Print stats & Calculate Class Weights (Requirement: Address class imbalance)
    print("\n--- Class Distribution ---")
    counts = df['diagnosis'].value_counts().sort_index()
    print(counts)

    # Calculate weights to balance the loss function later
    weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(df['diagnosis']),
        y=df['diagnosis']
    )
    class_weights = dict(enumerate(weights))
    print("\n--- Calculated Class Weights (for training) ---")
    print(class_weights)
    
    # 4. Visualization: Class Imbalance
    plt.figure(figsize=(10, 6))
    sns.countplot(x='diagnosis_name', data=df, hue='diagnosis_name', palette='viridis', order=diagnosis_lookup.values(), legend=False)
    plt.title('Severity Level Distribution (Class Imbalance)')
    plt.savefig('class_distribution.png')
    print("\n[Saved] class_distribution.png")

    # 5. Visualize 5 samples (Checking file existence first)
    plt.figure(figsize=(15, 8))
    for i in range(5):
        sample = df[df['diagnosis'] == i].iloc[0]
        img_path = os.path.join(img_dir, f"{sample['id_code']}.png")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(diagnosis_lookup[i])
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_images.png')
    print("[Saved] sample_images.png")
    
    return df

if __name__ == "__main__":
    if not os.path.exists("data/raw/train.csv"):
        print("Error: train.csv not found in data/raw/")
    else:
        run_eda()