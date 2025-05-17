from ultralytics import YOLO
from PIL import Image

# Load the trained model
model = YOLO("runs/classify/train2/weights/best.pt")  # Adjust path if needed

# Set the image path (update this with your actual image file)
image_path = "a.jpg"  # Replace with your image path

# Load and preprocess the image
image = Image.open(image_path).convert("RGB")

# Run prediction
results = model(image)

# Get the top predicted class
predicted_class = results[0].probs.top1
class_name = results[0].names[predicted_class]

print(f"Predicted Class: {class_name}")
