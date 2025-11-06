#!/usr/bin/env python3
"""
Lung Cancer Classification Web App
Upload images and get predictions with confidence scores
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw
import io
import base64
import os
import numpy as np
import cv2

# Import model architecture
from run_lung_cancer_model import CSPDarkNetSmall

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
MODEL_PATH = 'best_lung_cancer_model.pth'
CLASS_NAMES = ['Benign cases', 'Malignant cases', 'Normal cases']

model = None

def load_model():
    """Load the trained model"""
    global model
    if model is None:
        model = CSPDarkNetSmall(num_classes=len(CLASS_NAMES)).to(device)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        model.eval()
        print(f"✅ Model loaded from {MODEL_PATH}")
    return model

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def generate_gradcam(model, input_tensor, target_layer):
    """Generate Grad-CAM heatmap"""
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks
    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    
    # Forward pass
    output = model(input_tensor)
    pred_class = output.argmax(dim=1)
    
    # Backward pass
    model.zero_grad()
    output[0, pred_class].backward()
    
    # Remove hooks
    handle_forward.remove()
    handle_backward.remove()
    
    # Calculate Grad-CAM
    gradients_val = gradients[0].cpu().data.numpy()[0]
    activations_val = activations[0].cpu().data.numpy()[0]
    
    weights = np.mean(gradients_val, axis=(1, 2))
    cam = np.zeros(activations_val.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * activations_val[i]
    
    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - np.min(cam)
    cam = cam / (np.max(cam) + 1e-8)
    
    return cam

def detect_bounding_boxes(heatmap, threshold=0.5):
    """Detect bounding boxes from heatmap"""
    # Threshold the heatmap
    binary = (heatmap > threshold).astype(np.uint8) * 255
    
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Filter small regions
            x, y, w, h = cv2.boundingRect(contour)
            boxes.append((x, y, w, h))
    
    return boxes

def draw_boxes_on_image(image, boxes, predicted_class):
    """Draw bounding boxes on image"""
    draw = ImageDraw.Draw(image)
    
    # Choose color based on prediction
    if 'Malignant' in predicted_class:
        box_color = 'red'
        label = 'Cancer Detected'
    elif 'Benign' in predicted_class:
        box_color = 'yellow'
        label = 'Benign Lesion'
    else:
        return image  # No boxes for normal cases
    
    for (x, y, w, h) in boxes:
        # Draw rectangle
        draw.rectangle([x, y, x+w, y+h], outline=box_color, width=3)
        
        # Draw label
        draw.text((x, y-15), label, fill=box_color)
    
    return image

def predict_image(image):
    """Make prediction on uploaded image with bounding boxes"""
    model = load_model()
    
    # Preprocess image
    original_image = image.convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    
    # Get all class probabilities
    all_probs = probs[0].cpu().numpy()
    predicted_class = CLASS_NAMES[pred_class.item()]
    
    # Generate Grad-CAM for malignant or benign cases
    annotated_image = None
    if 'Malignant' in predicted_class or 'Benign' in predicted_class:
        # Get the last convolutional layer
        target_layer = model.stage4.concat_conv.conv
        
        # Generate heatmap
        heatmap = generate_gradcam(model, input_tensor, target_layer)
        
        # Detect bounding boxes
        boxes = detect_bounding_boxes(heatmap, threshold=0.6)
        
        # Draw boxes on original image
        annotated_image = original_image.copy()
        annotated_image = annotated_image.resize((224, 224))
        annotated_image = draw_boxes_on_image(annotated_image, boxes, predicted_class)
    
    return {
        'predicted_class': predicted_class,
        'confidence': confidence.item() * 100,
        'all_probabilities': {
            CLASS_NAMES[i]: float(all_probs[i] * 100) 
            for i in range(len(CLASS_NAMES))
        },
        'annotated_image': annotated_image
    }

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type. Please upload PNG or JPG'}), 400
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Make prediction
        result = predict_image(image)
        
        # Convert original image to base64
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        result['image'] = f"data:image/jpeg;base64,{image_base64}"
        
        # Convert annotated image to base64 if available
        if result['annotated_image']:
            buffered = io.BytesIO()
            result['annotated_image'].save(buffered, format="JPEG")
            annotated_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
            result['annotated_image'] = f"data:image/jpeg;base64,{annotated_base64}"
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Error processing image: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        exit(1)
    
    # Load model on startup
    load_model()
    
    print("\n" + "=" * 60)
    print("🫁 Lung Cancer Classification Web App")
    print("=" * 60)
    print(f"✅ Model loaded successfully")
    print(f"✅ Classes: {CLASS_NAMES}")
    print(f"✅ Device: {device}")
    print("\n🌐 Starting server...")
    print("📱 Open http://localhost:5001 in your browser")
    print("=" * 60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5001)
