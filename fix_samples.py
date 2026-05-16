import pandas as pd
import cv2
import matplotlib.pyplot as plt
import os

def generate_new_samples():
    df = pd.read_csv("data/raw/train.csv")
    img_dir = "data/raw/"
    
    diagnosis_lookup = {
        0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"
    }

    plt.figure(figsize=(20, 4))
    
    found_count = 0
    for i in range(5):
        # Get all IDs for this class
        class_ids = df[df['diagnosis'] == i]['id_code'].values
        
        # Try IDs until we find one that actually exists on disk
        for img_id in class_ids:
            img_path = os.path.join(img_dir, f"{img_id}.png")
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.subplot(1, 5, i+1)
                    plt.imshow(img)
                    plt.title(f"Class {i}: {diagnosis_lookup[i]}", fontsize=12)
                    plt.axis('off')
                    found_count += 1
                    break # Move to next class
                    
    if found_count > 0:
        plt.tight_layout()
        plt.savefig('sample_images_fixed.png')
        print("SUCCESS: sample_images_fixed.png generated!")
    else:
        print("ERROR: No images found. Check your data/raw folder.")

if __name__ == "__main__":
    generate_new_samples()