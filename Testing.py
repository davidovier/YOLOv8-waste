import os
import cv2
import pandas as pd
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="2CB9ynaTGGWwPR9Q0Mz0")
project = rf.workspace().project("waste-hsysm")
model = project.version(3).model

# Directory paths
input_dir = "/Users/davidvos/Desktop/Thesis/images"
output_dir = "predictions_made"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# List to store all prediction data
all_predictions = []

# Define a color dictionary for the six classes
class_colors = {
    "class1": (0, 0, 255),    # Red
    "class2": (0, 255, 0),    # Green
    "class3": (255, 0, 0),    # Blue
    "class4": (0, 255, 255),  # Yellow
    "class5": (255, 0, 255),  # Magenta
    "class6": (255, 255, 0)   # Cyan
}


# Iterate over each image in the folder
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    
    # Predict on the image
    prediction = model.predict(img_path, confidence=40, overlap=30).json()
    
    # Add prediction data to the list
    all_predictions.append(prediction)

    # Load image using OpenCV
    img = cv2.imread(img_path)

    # Draw bounding boxes and class labels on the image from the prediction JSON
    for item in prediction['predictions']:
        x = item['x']
        y = item['y']
        width = item['width']
        height = item['height']
        class_name = item['class']

        # Convert center coordinates (x, y) and width, height to top-left and bottom-right coordinates
        x1 = int(x - width / 2)
        y1 = int(y - height / 2)
        x2 = int(x + width / 2)
        y2 = int(y + height / 2)

        # Get the color for the class
        color = class_colors.get(class_name, (0, 255, 0))  # Default to green if class not found

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)  # Draw rectangle with the class color
        cv2.putText(img, class_name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Save the modified image
    cv2.imwrite(os.path.join(output_dir, img_name), img)

# Convert predictions list to DataFrame and save as CSV
df = pd.DataFrame(all_predictions)
df.to_csv('predictions.csv', index=False)
