# Quick Start Guide

## 🚀 Start the Web App

Run this command:
```bash
python app.py
```

Then open your browser to: **http://localhost:5000**

## 📸 How to Use

1. **Upload Image**: Click the upload area or drag & drop a lung scan image
2. **Analyze**: Click the "Analyze Image" button
3. **View Results**: See the prediction with confidence scores and probability breakdown

## 🎯 What You'll See

- **Predicted Class**: Benign, Malignant, or Normal
- **Confidence Score**: How confident the model is (0-100%)
- **Probability Breakdown**: Detailed percentages for all three classes

## 📊 Example Results

```
Predicted: Malignant cases
Confidence: 99.28%

Detailed Probabilities:
- Benign cases:    0.15%
- Malignant cases: 99.28%
- Normal cases:    0.57%
```

## ⚠️ Important Notes

- The model has **99% accuracy** for detecting malignant cases
- Lower accuracy for distinguishing benign vs normal cases
- Always consult medical professionals for actual diagnosis
- This is an AI tool for research/educational purposes

## 🛑 Stop the Server

Press `Ctrl+C` in the terminal where the app is running

## 🔧 Troubleshooting

**Can't access the app?**
- Make sure the server is running (you should see "Running on http://0.0.0.0:5000")
- Try http://127.0.0.1:5000 instead

**Upload not working?**
- Check file format (PNG, JPG, JPEG only)
- File size must be under 16MB

**Model error?**
- Ensure `best_lung_cancer_model.pth` is in the same folder as `app.py`
