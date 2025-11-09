# 🫁 Lung Cancer Detection System

An advanced deep learning-based web application for automated lung cancer detection and classification using state-of-the-art CSPDarkNet neural network architectures.

## 🎯 Overview

This system leverages multiple deep neural network architectures to classify lung CT scan images into three categories:
- **Benign cases**: Non-cancerous abnormalities
- **Malignant cases**: Cancerous tumors
- **Normal cases**: Healthy lung tissue

The application provides real-time predictions with confidence scores and visual explanations using Grad-CAM heatmaps and bounding box detection for identified abnormalities.

## ✨ Key Features

- **Multiple Neural Network Architectures**: Trained and compared CSPDarkNet53, CSPDarkNet101, and CSPDarkNetSmall models
- **Web-Based Interface**: User-friendly Flask application with authentication system
- **Visual Explanations**: Grad-CAM heatmaps and bounding boxes highlight regions of interest
- **Real-Time Predictions**: Instant classification with confidence scores for all classes
- **Secure Authentication**: User registration and login system with encrypted passwords
- **Comprehensive Metrics**: Detailed performance analysis including accuracy, precision, recall, and F1 scores
- **Production Ready**: Optimized for deployment with health check endpoints

## 🧠 Neural Network Architectures

### 1. CSPDarkNet53
- **Parameters**: ~27M
- **Architecture**: 5 stages with 1-2-8-8-4 residual blocks
- **Best for**: Balance between accuracy and speed
- **Training**: 50 epochs with data augmentation

### 2. CSPDarkNet101
- **Parameters**: ~45M
- **Architecture**: 5 stages with 2-4-23-23-5 residual blocks
- **Best for**: Maximum accuracy with deeper feature extraction
- **Training**: 50 epochs with dropout regularization

### 3. CSPDarkNetSmall
- **Parameters**: ~8M
- **Architecture**: 4 stages with 1-2-2-1 residual blocks
- **Best for**: Fast inference on resource-constrained devices
- **Training**: 30 epochs optimized for efficiency

## 🏗️ Architecture Highlights

All models implement:
- **CSP (Cross Stage Partial) connections** for efficient gradient flow
- **Residual blocks** to prevent vanishing gradients
- **Batch normalization** for training stability
- **LeakyReLU activation** for better gradient propagation
- **Adaptive pooling** for flexible input sizes
- **Class-weighted loss** to handle imbalanced datasets

## 📊 Model Performance

The models were trained on a curated dataset of lung CT scans with comprehensive evaluation:

- Confusion matrices for error analysis
- Per-class precision, recall, and F1 scores
- Training/validation loss and accuracy curves
- Learning rate schedules
- Detailed classification reports

Results and visualizations are saved in respective model directories.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd lung-cancer-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your dataset:
```
dataset/
├── Bengin cases/
│   ├── image1.jpg
│   └── ...
├── Malignant cases/
│   ├── image1.jpg
│   └── ...
└── Normal cases/
    ├── image1.jpg
    └── ...
```

### Training Models

Train CSPDarkNet53:
```bash
cd "cspdarknet53 n 101"
python train_cspdarknet53.py
```

Train CSPDarkNet101:
```bash
cd "cspdarknet53 n 101"
python train_cspdarknet101.py
```

Train the base model:
```bash
python run_lung_cancer_model.py
```

### Running the Web Application

1. Ensure you have a trained model file (`cspdarknet53_best.pth`)

2. Start the Flask server:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5001
```

4. Create an account and start uploading CT scan images for analysis

## 🖥️ Web Interface

### Features:
- **User Authentication**: Secure signup/login system
- **Image Upload**: Drag-and-drop or click to upload CT scans
- **Real-Time Analysis**: Instant predictions with confidence scores
- **Visual Feedback**: 
  - Original image display
  - Annotated image with bounding boxes (for malignant/benign cases)
  - Probability distribution across all classes
- **Responsive Design**: Works on desktop and mobile devices

## 📁 Project Structure

```
lung-cancer-detection/
├── app.py                          # Flask web application
├── auth.py                         # Authentication system
├── run_lung_cancer_model.py        # Base model training script
├── requirements.txt                # Python dependencies
├── cspdarknet53_best.pth          # Trained model weights
├── users.db                        # User database
│
├── cspdarknet53 n 101/            # Advanced model experiments
│   ├── train_cspdarknet53.py      # CSPDarkNet53 training
│   ├── train_cspdarknet101.py     # CSPDarkNet101 training
│   ├── plot_comparison.py         # Model comparison tools
│   ├── cspdarknet53_results/      # Training results & metrics
│   └── cspdarknet101_results/     # Training results & metrics
│
├── cspdarknetsmall/               # Lightweight model variant
│   ├── analyze_model.py           # Model analysis tools
│   └── test_results.txt           # Test predictions
│
├── dataset/                        # Training data
│   ├── Bengin cases/
│   ├── Malignant cases/
│   └── Normal cases/
│
├── test images/                    # Test dataset
│   ├── Bengin case (2).jpg
│   ├── Malignant case (2).jpg
│   └── Normal case (13).jpg
│
└── templates/                      # HTML templates
    ├── index.html                  # Main application page
    ├── login.html                  # Login page
    └── signup.html                 # Registration page
```

## 🔬 Technical Details

### Data Preprocessing
- Image resizing to 224×224 pixels
- Normalization with mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
- Data augmentation:
  - Random horizontal flips
  - Random rotation (±15°)
  - Random resized crops
  - Color jittering

### Training Strategy
- **Optimizer**: Adam with learning rate 0.0003-0.0005
- **Loss Function**: Cross-entropy with class weights
- **Scheduler**: StepLR (decay every 15 epochs)
- **Batch Size**: 8-16 (depending on model size)
- **Validation Split**: 80/20 train-validation split

### Grad-CAM Visualization
The system uses Gradient-weighted Class Activation Mapping (Grad-CAM) to:
- Highlight regions contributing to predictions
- Generate heatmaps showing model attention
- Detect and draw bounding boxes around abnormalities
- Provide interpretable AI decisions

## 📈 Model Comparison

Use the comparison tools to evaluate different architectures:

```bash
cd "cspdarknet53 n 101"
python plot_comparison.py
```

This generates:
- Side-by-side accuracy comparisons
- Loss curve comparisons
- Per-class performance metrics
- Inference time analysis

## 🔒 Security Features

- Password hashing using Werkzeug security
- Flask-Login session management
- CSRF protection
- Secure file upload validation
- Maximum file size limits (16MB)
- SQL injection prevention

## 🧪 Testing

Test the model on new images:

```bash
python run_lung_cancer_model.py
```

Or use the web interface to upload individual images for real-time predictions.

## 📊 Performance Metrics

The system tracks and reports:
- **Accuracy**: Overall classification correctness
- **Precision**: True positive rate per class
- **Recall**: Sensitivity per class
- **F1 Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed error analysis
- **ROC Curves**: Classification performance visualization

## 🛠️ API Endpoints

- `GET /`: Main application page (requires authentication)
- `POST /signup`: User registration
- `POST /login`: User authentication
- `GET /logout`: User logout
- `POST /predict`: Image classification endpoint
- `GET /health`: Health check for monitoring

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:
- Bug fixes
- New features
- Documentation improvements
- Performance optimizations
- Additional model architectures

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- CSPDarkNet architecture inspired by YOLOv4
- Grad-CAM implementation for visual explanations
- Flask framework for web application
- PyTorch for deep learning capabilities

## ⚠️ Disclaimer

This system is designed for research and educational purposes. It should not be used as a substitute for professional medical diagnosis. Always consult qualified healthcare professionals for medical advice and diagnosis.

## 🔮 Future Enhancements

- [ ] Multi-modal input support (CT + clinical data)
- [ ] 3D CNN for volumetric analysis
- [ ] Ensemble model predictions
- [ ] REST API for integration
- [ ] Docker containerization
- [ ] Cloud deployment guides
- [ ] Mobile application
- [ ] Real-time video stream analysis
- [ ] Explainable AI dashboard
- [ ] Multi-language support

---


