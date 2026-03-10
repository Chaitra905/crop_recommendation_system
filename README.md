🌾 Crop Recommendation Using Soil Image Processing and Machine Learning
📌 Project Overview

Agriculture plays a crucial role in economic growth and food security. Selecting the right crop based on soil type is essential for improving productivity and sustainability. Traditional soil testing methods are often time-consuming, expensive, and not easily accessible for many farmers.

This project presents a Machine Learning-based Crop Recommendation System that analyzes soil images and predicts the soil type using deep learning techniques. Based on the predicted soil category, the system recommends suitable crops that can grow efficiently in that soil.

The system uses image processing and a Convolutional Neural Network (CNN) to extract visual features such as color, texture, and surface patterns from soil images. This approach eliminates the need for laboratory soil testing and enables quick decision-making.

🚀 Features

Soil classification using deep learning

Crop recommendation based on soil type

Image preprocessing for better model accuracy

User-friendly web interface

Real-time soil image prediction

Easy deployment using Flask

🛠 Technologies Used

Programming Language

Python 3

Machine Learning

TensorFlow

Keras

CNN (Convolutional Neural Network)

Web Development

Flask

HTML

CSS

JavaScript

Libraries

NumPy

OpenCV

Pillow

📂 Project Structure
Crop_Recommendation_ML
│
├── app.py
├── load_and_train.py
├── predict_image.py
├── evaluate_model.py
├── check_accuracy.py
│
├── models
│   ├── soil_model.h5
│   └── soil_model_balanced.keras
│
├── templates
│   └── index.html
│
├── static
│   ├── css
│   ├── js
│   └── crops
│
├── datasets
│
└── results
⚙️ How It Works

1️⃣ User uploads a soil image
2️⃣ Image is preprocessed (resize, normalize, noise reduction)
3️⃣ CNN model extracts features from the image
4️⃣ Model predicts soil type
5️⃣ System recommends suitable crops for that soil

📊 Model Performance

Evaluation metrics used:

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

The trained model achieves approximately 85–90% accuracy depending on dataset quality.

▶️ How to Run the Project
1️⃣ Clone the Repository
git clone https://github.com/Chaitra905/crop_recommendation_system.git
2️⃣ Navigate to Project Folder
cd crop_recommendation_system
3️⃣ Create Virtual Environment
python -m venv ml_env
4️⃣ Activate Environment

Windows

ml_env\Scripts\activate
5️⃣ Install Dependencies
pip install tensorflow flask numpy pillow opencv-python
6️⃣ Run Application
python app.py
7️⃣ Open Browser
http://127.0.0.1:5000
📸 Application Workflow
Soil Image Upload
        ↓
Image Preprocessing
        ↓
CNN Model Prediction
        ↓
Soil Type Classification
        ↓
Crop Recommendation
        ↓
Display Results
🔮 Future Improvements

Weather-based crop prediction

Fertilizer recommendation system

Mobile application integration

Multilingual farmer support

Larger soil dataset for improved accuracy

📚 References

Kaggle Soil Dataset

TensorFlow Documentation

Research papers on soil classification and crop recommendation

Machine Learning and Deep Learning resources
