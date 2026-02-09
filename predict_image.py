import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------ CONFIG ------------------
MODEL_PATH = "crop_recommendation_model.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)        # ✅ FIXED
CONFIDENCE_THRESHOLD = 0.70
# --------------------------------------------

# Load model
model = load_model(MODEL_PATH)
print("✅ Model loaded")

# Load class names
with open(CLASS_NAMES_PATH, "r") as f:
    class_names = json.load(f)

print("📂 Classes:", class_names)

# Ask user for image path
img_path = input("\nEnter full path of the image to predict: ").strip()

# Validate path
if not os.path.exists(img_path):
    print("❌ Image not found")
    exit()

# Load & preprocess image
try:
    img = Image.open(img_path).convert("RGB")
    img = img.resize(IMG_SIZE)   # ✅ 224x224
except Exception:
    print("❌ Invalid image file")
    exit()

img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
predictions = model.predict(img_array)
confidence = float(np.max(predictions))
predicted_index = int(np.argmax(predictions))

# Output
print("\n🌱 Prediction Result")
print("--------------------")

if confidence < CONFIDENCE_THRESHOLD:
    print("❌ This image is NOT a soil image")
    print(f"📊 Confidence : {confidence * 100:.2f}%")
else:
    print(f"✅ Soil Type  : {class_names[predicted_index]}")
    print(f"📊 Confidence: {confidence * 100:.2f}%")
