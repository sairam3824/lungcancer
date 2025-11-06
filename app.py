#!/usr/bin/env python3
"""
Lung Cancer Classification Web App
Upload images and get predictions with confidence scores
"""

from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import os

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

def predict_image(image):
    """Make prediction on uploaded image"""
    model = load_model()
    
    # Preprocess image
    image = image.convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, 1)
    
    # Get all class probabilities
    all_probs = probs[0].cpu().numpy()
    
    return {
        'predicted_class': CLASS_NAMES[pred_class.item()],
        'confidence': confidence.item() * 100,
        'all_probabilities': {
            CLASS_NAMES[i]: float(all_probs[i] * 100) 
            for i in range(len(CLASS_NAMES))
        }
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
        
        # Convert image to base64 for display
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        result['image'] = f"data:image/jpeg;base64,{image_base64}"
        
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
