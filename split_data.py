import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Define paths
base_dir = r"archive (4)\data\food-101-tiny"  # adjust if needed
source_dir = os.path.join(base_dir, "train")
train_output_dir = os.path.join(base_dir, "train")
test_output_dir = os.path.join(base_dir, "test")

# Create test directory if not exists
if not os.path.exists(test_output_dir):
    os.makedirs(test_output_dir)

# Split ratio (e.g., 80% train, 20% test)
split_ratio = 0.8

# Loop over each class
for class_name in os.listdir(source_dir):
    class_path = os.path.join(source_dir, class_name)
    
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    test_images = images[split_index:]

    # Create test class folder
    test_class_dir = os.path.join(test_output_dir, class_name)
    os.makedirs(test_class_dir, exist_ok=True)

    # Move test images
    for image in test_images:
        src_path = os.path.join(class_path, image)
        dst_path = os.path.join(test_class_dir, image)
        shutil.copy(src_path, dst_path)  # use copy to retain original in train
