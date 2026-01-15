# DermAI

A web application for dermatological disease classification using ResNet 50 CNN model. This application uses a tweaked trained model to classify skin conditions from uploaded images.

## Features

- Upload skin condition images for classification
- Get predictions for 10 different dermatological conditions
- View detailed information about each condition including:
  - Description
  - Treatment recommendations
  - When to consult a dermatologist
  - Urgency level

## Supported Conditions

1. Eczema
2. Warts Molluscum and other Viral Infections
3. Melanoma
4. Atopic Dermatitis
5. Basal Cell Carcinoma (BCC)
6. Melanocytic Nevi (NV)
7. Benign Keratosis-like Lesions (BKL)
8. Psoriasis pictures Lichen Planus and related diseases
9. Seborrheic Keratoses and other Benign Tumors
10. Tinea Ringworm Candidiasis and other Fungal Infections

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd DermAI
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Ensure you have the model file `DermAI.keras` in the project root directory.

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

3. Upload a skin condition image and get predictions.

## Model

The application uses a pre-trained Keras model (`DermAI.keras`) based on ResNet50 architecture. The model processes images at 224x224 resolution and provides confidence scores for each condition.

## Important Notes

**Medical Disclaimer**: This application is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified dermatologist for accurate diagnosis and treatment of skin conditions.

## Requirements

See `requirements.txt` for the complete list of dependencies. Main dependencies include:
- Flask 3.1.2
- TensorFlow 2.20.0
- Keras 3.10.0
- Pillow 12.0.0
- NumPy 2.3.5


