import os
import shutil
import pandas as pd

# 1. Define paths
base_dir = "data/raw"
csv_path = os.path.join(base_dir, "train.csv")

# 2. Find all images in any subfolder
print("Searching for images...")
found_images = {}
for root, dirs, files in os.walk(base_dir):
    for file in files:
        if file.endswith(".png"):
            found_images[file] = os.path.join(root, file)

print(f"Found {len(found_images)} images.")

# 3. Move images to data/raw/
for filename, full_path in found_images.items():
    destination = os.path.join(base_dir, filename)
    if full_path != destination:
        shutil.move(full_path, destination)

# 4. Verify CSV vs Files
df = pd.read_csv(csv_path)
sample_id = df['id_code'].iloc[0]
expected_file = os.path.join(base_dir, f"{sample_id}.png")

if os.path.exists(expected_file):
    print(f"SUCCESS: Found {sample_id}.png in {base_dir}")
else:
    print(f"STILL MISSING: Looked for {expected_file}")
    # List first 5 files actually in the folder to see what happened
    print("Files actually in data/raw:", os.listdir(base_dir)[:5])