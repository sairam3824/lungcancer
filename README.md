# 🫁 Lung Cancer Classification System

An AI-powered deep learning system for classifying lung cancer from medical images using a custom CSPDarkNet architecture. Features a web interface with real-time predictions and visual cancer region detection.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## ✨ Features

- **High Accuracy**: 99%+ accuracy for malignant case detection
- **Custom Architecture**: CSPDarkNet-based neural network optimized for medical imaging
- **Visual Detection**: Grad-CAM powered bounding boxes highlight cancer regions
- **Web Interface**: User-friendly Flask app with drag-and-drop upload
- **Real-time Analysis**: Instant predictions with confidence scores
- **Comprehensive Metrics**: Detailed performance analysis with confusion matrices

## 🎯 Classification Categories

- **Benign Cases**: Non-cancerous abnormalities
- **Malignant Cases**: Cancerous tumors
- **Normal Cases**: Healthy lung tissue

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify the model file exists:
```bash
ls best_lung_cancer_model.pth
```

### Running the Web Application

Start the Flask server:
```bash
python app.py
```

Open your browser to: **http://localhost:5001**

### Using the Application

1. Upload a lung scan image (PNG, JPG, or JPEG)
2. Click "Analyze Image"
3. View prediction results with:
   - Predicted class
   - Confidence score
   - Probability breakdown
   - Visual bounding boxes (for malignant/benign cases)

## 📊 Model Performance

The CSPDarkNet model achieves excellent performance across all classes:

- **Overall Accuracy**: 99%+
- **Malignant Detection**: 99%+ precision and recall
- **Benign Detection**: High accuracy with visual localization
- **Normal Detection**: Reliable classification

See `model_analysis_report.txt` for detailed metrics.

## 🏗️ Architecture

### CSPDarkNet Model

The system uses a custom CSPDarkNet (Cross Stage Partial DarkNet) architecture:

- **Input**: 224x224 RGB images
- **Backbone**: CSP blocks with residual connections
- **Stages**: 4 progressive downsampling stages
- **Output**: 3-class softmax classification

Key components:
- Convolutional blocks with batch normalization
- Residual connections for gradient flow
- CSP blocks for efficient feature extraction
- Global average pooling for classification

### Grad-CAM Visualization

For malignant and benign cases, the system generates:
- Heatmaps showing regions of interest
- Bounding boxes highlighting detected abnormalities
- Color-coded annotations (red for malignant, yellow for benign)

## 📁 Project Structure

```
.
├── app.py                          # Flask web application
├── run_lung_cancer_model.py        # Model training script
├── analyze_model.py                # Performance analysis tool
├── best_lung_cancer_model.pth      # Trained model weights
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                  # Web interface
├── dataset/                        # Training data
│   ├── Bengin cases/
│   ├── Malignant cases/
│   └── Normal cases/
├── test images/                    # Test samples
└── *.png                          # Generated visualizations
```

## 🔧 Advanced Usage

### Training a New Model

To train the model from scratch:

```bash
python run_lung_cancer_model.py
```

This will:
- Load images from the `dataset/` folder
- Train for 30 epochs with data augmentation
- Save the best model as `best_lung_cancer_model.pth`
- Test on images in `test images/` folder

### Analyzing Model Performance

Generate comprehensive metrics and visualizations:

```bash
python analyze_model.py
```

Outputs:
- `model_analysis_report.txt` - Detailed metrics
- `confusion_matrix.png` - Confusion matrix visualization
- `class_distribution.png` - Dataset distribution
- `per_class_metrics.png` - Per-class performance

### API Usage

The Flask app provides a REST API:

**POST /predict**
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5001/predict
```

Response:
```json
{
  "predicted_class": "Malignant cases",
  "confidence": 99.28,
  "all_probabilities": {
    "Benign cases": 0.15,
    "Malignant cases": 99.28,
    "Normal cases": 0.57
  },
  "image": "data:image/jpeg;base64,...",
  "annotated_image": "data:image/jpeg;base64,..."
}
```

## 🛠️ Configuration

### Model Parameters

Edit `run_lung_cancer_model.py` to adjust:
- `num_epochs`: Training duration (default: 30)
- `batch_size`: Batch size (default: 16)
- `learning_rate`: Initial learning rate (default: 0.0005)

### Data Augmentation

Training uses:
- Random resized crop (80-100% scale)
- Random horizontal flip (50% probability)
- Random rotation (±15 degrees)
- Color jitter (brightness, contrast, saturation)

### Web Server

Edit `app.py` to change:
- `port`: Server port (default: 5001)
- `MAX_CONTENT_LENGTH`: Max upload size (default: 16MB)
- `host`: Bind address (default: 0.0.0.0)

## 📈 Performance Metrics

The model is evaluated using:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/true positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

## ⚠️ Important Disclaimers

- **Medical Use**: This system is for research and educational purposes only
- **Not FDA Approved**: Not intended for clinical diagnosis
- **Consult Professionals**: Always seek professional medical advice
- **No Warranty**: Provided as-is without guarantees

## 🔬 Technical Details

### Dependencies

- **PyTorch**: Deep learning framework
- **torchvision**: Image transformations
- **Flask**: Web framework
- **OpenCV**: Image processing and Grad-CAM
- **scikit-learn**: Metrics and evaluation
- **matplotlib/seaborn**: Visualization
- **Pillow**: Image handling

### Hardware Requirements

- **Minimum**: CPU with 4GB RAM
- **Recommended**: NVIDIA GPU with CUDA support
- **Storage**: 500MB for model and dependencies

### Dataset Format

Images should be organized as:
```
dataset/
  ├── Bengin cases/
  │   ├── image1.jpg
  │   └── image2.jpg
  ├── Malignant cases/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── Normal cases/
      ├── image1.jpg
      └── image2.jpg
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional data augmentation techniques
- Model architecture enhancements
- UI/UX improvements
- Multi-language support
- Mobile app development

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CSPDarkNet architecture inspired by YOLOv4
- Grad-CAM implementation for medical image visualization
- Flask framework for rapid web development

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

## 🔗 References

- [CSPNet Paper](https://arxiv.org/abs/1911.11929)
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

**Note**: This is an AI-assisted diagnostic tool for research purposes. Always consult qualified healthcare professionals for medical decisions.
