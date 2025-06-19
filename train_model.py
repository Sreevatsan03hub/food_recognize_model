import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ✅ Update these paths to match your system structure
train_dir = r'C:\Users\srees\OneDrive\Documents\Desktop\food recognize model\archive (4)\data\food-101-tiny\train'
test_dir = r'C:\Users\srees\OneDrive\Documents\Desktop\food recognize model\archive (4)\data\food-101-tiny\test'

# Image settings
img_size = (128, 128)
batch_size = 32

# Data loading with augmentation
train_gen = ImageDataGenerator(rescale=1./255, zoom_range=0.2, horizontal_flip=True)
test_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

test_data = test_gen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(train_data.num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(train_data, validation_data=test_data, epochs=10)

# Save model
model.save("food_model.h5")
print("✅ Model trained and saved as food_model.h5")
