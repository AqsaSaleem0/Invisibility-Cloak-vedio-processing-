import os

# Set the folder path containing JPG images
folder_path = "D:/invisibility_cloak/dataset"

# Get all JPG files in the folder
jpg_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".jpg")]
jpg_files.sort()  # Sort files to maintain order

# Rename JPG images sequentially
for index, filename in enumerate(jpg_files, start=1):
    old_path = os.path.join(folder_path, filename)
    new_filename = f"image_{index}.jpg"  # New naming format
    new_path = os.path.join(folder_path, new_filename)
    
    os.rename(old_path, new_path)
    print(f"Renamed: {filename} -> {new_filename}")

print("Renaming completed!")
