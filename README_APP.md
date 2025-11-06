# Lung Cancer Classification Web App

A Flask-based web application for classifying lung cancer images using the trained CSPDarkNet model.

## Features

- 🖼️ Drag-and-drop or click to upload lung scan images
- 🤖 Real-time AI-powered classification
- 📊 Detailed probability breakdown for all classes
- 🎨 Beautiful, responsive UI
- ⚡ Fast predictions

## Installation

1. Install Flask (if not already installed):
```bash
pip install flask
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

## Running the App

1. Make sure you have the trained model file `best_lung_cancer_model.pth` in the same directory

2. Start the Flask server:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. Click the upload area or drag and drop a lung scan image (PNG, JPG, or JPEG)
2. Click "Analyze Image" button
3. View the prediction results with confidence scores
4. See detailed probability breakdown for all three classes:
   - Benign cases
   - Malignant cases
   - Normal cases

## API Endpoints

### GET /
Main web interface

### POST /predict
Upload and classify an image

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response:**
```json
{
  "predicted_class": "Malignant cases",
  "confidence": 99.28,
  "all_probabilities": {
    "Benign cases": 0.15,
    "Malignant cases": 99.28,
    "Normal cases": 0.57
  }
}
```

### GET /health
Health check endpoint

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## File Structure

```
.
├── app.py                          # Flask application
├── templates/
│   └── index.html                  # Web interface
├── best_lung_cancer_model.pth      # Trained model
├── run_lung_cancer_model.py        # Model architecture
└── requirements.txt                # Python dependencies
```

## Notes

- Maximum file size: 16MB
- Supported formats: PNG, JPG, JPEG
- The app runs on port 5000 by default
- Model predictions are based on the trained CSPDarkNet model

## Troubleshooting

**Model not found error:**
- Ensure `best_lung_cancer_model.pth` exists in the same directory as `app.py`

**Import error:**
- Make sure `run_lung_cancer_model.py` is in the same directory
- Install all required packages: `pip install -r requirements.txt`

**Port already in use:**
- Change the port in `app.py`: `app.run(port=5001)`
