import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load trained model
model = load_model("food_model.h5")

# Get class labels directly from your dataset folder
labels = sorted(os.listdir("archive (4)/data/food-101-tiny/train"))

# Updated calorie dictionary for these 10 classes
calorie_dict = {
    "apple_pie": 296,
    "cheesecake": 321,
    "chocolate_cake": 371,
    "donuts": 452,
    "french_fries": 312,
    "fried_rice": 163,
    "grilled_salmon": 206,
    "ice_cream": 207,
    "pizza": 266,
    "sushi": 200
}

def predict_food(img_path):
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    index = np.argmax(prediction)
    food_item = labels[index]
    confidence = np.max(prediction) * 100

    calories = calorie_dict.get(food_item.lower(), "Calorie info not available")

    print(f"\n Food Item: {food_item.replace('_', ' ').title()}")
    print(f" Confidence: {confidence:.2f}%")
    print(f" Estimated Calories: {calories} kcal\n")

# Example usage – make sure this path is correct
predict_food("archive (4)/data/food-101-tiny/train/apple_pie/112378.jpg")
