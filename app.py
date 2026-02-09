from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from PIL import Image

app = Flask(__name__)

# ==========================
# LOAD MODEL
# ==========================
MODEL_PATH = "crop_recommendation_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# ==========================
# CLASS NAMES (MUST MATCH TRAINING ORDER)
# ==========================
class_names = [
    "Alluvial_Soil",
    "Arid_Soil",
    "Black_Soil",
    "Laterite_Soil",
    "Mountain_Soil",
    "Red_Soil",
    "Yellow_Soil"
]

print("✅ Detected classes:", class_names)

# ==========================
# CROP RECOMMENDATION
# ==========================
crop_recommendation = {
    "Black_Soil": ["Cotton", "Soybean", "Wheat"],
    "Red_Soil": ["Millet", "Groundnut", "Pulses"],
    "Yellow_Soil": ["Maize", "Sunflower", "Potato"],
    "Alluvial_Soil": ["Paddy", "Sugarcane", "Tobacco"],
    "Laterite_Soil": ["Tea", "Coffee", "Cashew"],
    "Mountain_Soil": ["Apple", "Barley", "Tea"],
    "Arid_Soil": ["Tobacco", "Sesame", "Mung bean"]
}

# ==========================
# HOME PAGE
# ==========================
@app.route('/')
def home():
    return render_template('index.html')

# ==========================
# PREDICT ROUTE
# ==========================
@app.route('/predict', methods=['POST'])
def predict():

    if 'image' not in request.files or request.files['image'].filename == '':
        return render_template('index.html', result="⚠️ No file uploaded.")

    file = request.files['image']

    file_path = os.path.join('static', 'uploads', file.filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file.save(file_path)

    # ==========================
    # IMAGE PREPROCESSING
    # ==========================
    img = Image.open(file_path).convert("RGB").resize((224,224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    # ==========================
    # PREDICTION
    # ==========================
    predictions = model.predict(img_array)
    predicted_class = int(np.argmax(predictions))
    confidence = float(np.max(predictions))

    print("Confidence:", confidence)

    # ==========================
    # NON-SOIL IMAGE CHECK
    # ==========================
    if confidence < 0.80:
        return render_template(
            'index.html',
            result="❌ Please upload a proper SOIL image only!",
            crops="No crops available",
            image=file.filename
        )

    # ==========================
    # VALIDATE CLASS
    # ==========================
    if predicted_class >= len(class_names):
        return render_template(
            'index.html',
            result="⚠️ Prediction error: Invalid class index."
        )

    soil_type = class_names[predicted_class]
    print("Predicted Soil:", soil_type)

    crops = ", ".join(
        crop_recommendation.get(soil_type, ["No data available"])
    )

    return render_template(
        'index.html',
        result=soil_type,
        crops=crops,
        image=file.filename
    )

# ==========================
# RUN SERVER
# ==========================
if __name__ == "__main__":
    app.run(debug=True)
