from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from PIL import Image
import os

# Load your trained YOLO classification model
model = YOLO("runs/classify/train/weights/best.pt")

app = Flask(__name__)
CORS(app)  # Allow requests from Android or any frontend

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            print("No image in request!")  
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        image = Image.open(file).convert("RGB")
        print("Image received, running prediction...")

        results = model(image)
        predicted_class = results[0].probs.top1
        class_name = results[0].names[predicted_class]
        print(f"Predicted Class: {class_name}")

        return jsonify({'prediction': class_name})

    except Exception as e:
        print(f"Predicted Class: {class_name}")
       
    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)

