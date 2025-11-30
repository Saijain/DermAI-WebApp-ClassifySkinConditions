import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import os
import glob

model_path = 'DermAI.keras'

# Define class names based on training (Cell 18 from Colab notebook)
# Note: Eczema (class 0) and Atopic Dermatitis (class 3) are combined as they are the same condition
CLASS_NAMES = {
    0: 'Eczema/Atopic Dermatitis',
    1: 'Warts Molluscum and other Viral Infections',
    2: 'Melanoma',
    3: 'Eczema/Atopic Dermatitis',  # Combined with class 0
    4: 'Basal Cell Carcinoma (BCC)',
    5: 'Melanocytic Nevi (NV)',
    6: 'Benign Keratosis-like Lesions (BKL)',
    7: 'Psoriasis pictures Lichen Planus and related diseases',
    8: 'Seborrheic Keratoses and other Benign Tumors',
    9: 'Tinea Ringworm Candidiasis and other Fungal Infections'
}

# Combined class names for evaluation (9 classes instead of 10)
COMBINED_CLASS_NAMES = {
    0: 'Eczema/Atopic Dermatitis',
    1: 'Warts Molluscum and other Viral Infections',
    2: 'Melanoma',
    3: 'Basal Cell Carcinoma (BCC)',
    4: 'Melanocytic Nevi (NV)',
    5: 'Benign Keratosis-like Lesions (BKL)',
    6: 'Psoriasis pictures Lichen Planus and related diseases',
    7: 'Seborrheic Keratoses and other Benign Tumors',
    8: 'Tinea Ringworm Candidiasis and other Fungal Infections'
}

# Map original class indices to combined class indices
# Classes 0 and 3 both map to combined class 0
# Classes 1,2,4,5,6,7,8,9 map to combined classes 1,2,3,4,5,6,7,8
ORIGINAL_TO_COMBINED = {
    0: 0,  # Eczema -> Eczema/Atopic Dermatitis
    1: 1,  # Warts -> Warts
    2: 2,  # Melanoma -> Melanoma
    3: 0,  # Atopic Dermatitis -> Eczema/Atopic Dermatitis (combined with 0)
    4: 3,  # BCC -> BCC
    5: 4,  # NV -> NV
    6: 5,  # BKL -> BKL
    7: 6,  # Psoriasis -> Psoriasis
    8: 7,  # Seborrheic -> Seborrheic
    9: 8   # Tinea -> Tinea
}

# Map directory names to class indices (matching the training class order)
DIRECTORY_TO_CLASS = {
    '1. Eczema 1677': 0,
    '2. Melanoma 15.75k': 2,  # Changed from 1 to 2
    '3. Atopic Dermatitis - 1.25k': 3,  # Changed from 2 to 3
    '4. Basal Cell Carcinoma (BCC) 3323': 4,  # Changed from 3 to 4
    '5. Melanocytic Nevi (NV) - 7970': 5,  # Changed from 4 to 5
    '6. Benign Keratosis-like Lesions (BKL) 2624': 6,  # Changed from 5 to 6
    '7. Psoriasis pictures Lichen Planus and related diseases - 2k': 7,  # Changed from 6 to 7
    '8. Seborrheic Keratoses and other Benign Tumors - 1.8k': 8,  # Changed from 7 to 8
    '9. Tinea Ringworm Candidiasis and other Fungal Infections - 1.7k': 9,  # Changed from 8 to 9
    '10. Warts Molluscum and other Viral Infections - 2103': 1  # Changed from 9 to 1
}

def preprocess_image(image_path):
    """Preprocess a single image for prediction (matching training preprocessing from app.py)"""
    image = Image.open(image_path)
    
    # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = image.resize((224, 224))
    image = np.array(image)
    
    # Ensure image is 3-channel RGB (handles edge cases)
    if len(image.shape) == 2:  # Grayscale
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA
        image = image[:, :, :3]  # Remove alpha channel
    
    # CRITICAL: Use the same preprocessing as during training (ResNet50 ImageNet preprocessing)
    # This converts RGB to BGR and applies ImageNet mean normalization
    image = preprocess_input(image.astype('float32'))
    
    image = np.expand_dims(image, axis=0)
    return image

def evaluate_model(model, test_data_dir, num_samples_per_class=50):
    """
    Evaluate the model on test data
    
    Args:
        model: The loaded Keras model
        test_data_dir: Directory containing subdirectories for each class
        num_samples_per_class: Number of samples to test per class (None for all)
    """
    print("Loading test data...")
    
    y_true = []
    y_pred = []
    predictions_list = []
    file_paths = []
    
    # Iterate through each class directory
    for dir_name, class_idx in DIRECTORY_TO_CLASS.items():
        class_dir = os.path.join(test_data_dir, dir_name)
        
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found. Skipping...")
            continue
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(class_dir, ext)))
        
        if not image_files:
            print(f"Warning: No images found in {class_dir}")
            continue
        
        # Limit number of samples if specified
        if num_samples_per_class and len(image_files) > num_samples_per_class:
            import random
            image_files = random.sample(image_files, num_samples_per_class)
        
        # Get display name (use combined name if it's Eczema or Atopic Dermatitis)
        display_name = COMBINED_CLASS_NAMES[ORIGINAL_TO_COMBINED[class_idx]]
        print(f"Processing {len(image_files)} images from {display_name}...")
        
        # Process each image
        for image_path in image_files:
            try:
                # Preprocess image
                image = preprocess_image(image_path)
                
                # Get prediction
                prediction = model.predict(image, verbose=0)
                
                # Combine Eczema (class 0) and Atopic Dermatitis (class 3) predictions
                # Add their probabilities together and assign to class 0, set class 3 to 0
                combined_prediction = prediction[0].copy()
                eczema_atopic_combined = prediction[0][0] + prediction[0][3]
                combined_prediction[0] = eczema_atopic_combined
                combined_prediction[3] = 0.0  # Set to 0 so argmax won't pick it separately
                
                # Find predicted class (using combined prediction)
                predicted_class_idx = np.argmax(combined_prediction)
                
                # Map to combined class index for evaluation
                # ORIGINAL_TO_COMBINED already maps both 0 and 3 to combined class 0
                predicted_combined_idx = ORIGINAL_TO_COMBINED[predicted_class_idx]
                true_combined_idx = ORIGINAL_TO_COMBINED[class_idx]
                
                # Store results using combined indices
                y_true.append(true_combined_idx)
                y_pred.append(predicted_combined_idx)
                predictions_list.append(prediction[0])
                file_paths.append(image_path)
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue
    
    if len(y_true) == 0:
        print("No test samples found!")
        return
    
    # Calculate metrics
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate accuracy manually
    accuracy = np.mean(y_true == y_pred)
    
    # Build confusion matrix manually (using combined classes - 9 classes)
    num_classes = len(COMBINED_CLASS_NAMES)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label][pred_label] += 1
    
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"\nTotal test samples: {len(y_true)}")
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")
    
    # Per-class metrics
    print("\n" + "-"*60)
    print("PER-CLASS METRICS:")
    print("-"*60)
    print(f"{'Class':<50} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 100)
    
    for i in range(num_classes):
        class_name = COMBINED_CLASS_NAMES[i]
        true_positives = cm[i][i]
        false_positives = np.sum(cm[:, i]) - true_positives
        false_negatives = np.sum(cm[i, :]) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        support = np.sum(y_true == i)
        
        print(f"{class_name:<50} {precision:<12.4f} {recall:<12.4f} {f1_score:<12.4f} {support:<10}")
    
    # Confusion matrix
    print("\n" + "-"*60)
    print("CONFUSION MATRIX:")
    print("-"*60)
    
    # Print confusion matrix with labels
    print("\nActual\\Predicted", end="")
    for i in range(num_classes):
        print(f"\t{COMBINED_CLASS_NAMES[i][:15]}", end="")
    print()
    
    for i in range(num_classes):
        print(f"{COMBINED_CLASS_NAMES[i][:25]}", end="")
        for j in range(num_classes):
            print(f"\t{cm[i][j]}", end="")
        print()
    
    # Calculate per-class accuracy
    print("\n" + "-"*60)
    print("PER-CLASS ACCURACY:")
    print("-"*60)
    for i in range(num_classes):
        class_name = COMBINED_CLASS_NAMES[i]
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum((y_true == i) & (y_pred == i)) / np.sum(class_mask)
            print(f"{class_name}: {class_accuracy * 100:.2f}% ({np.sum((y_true == i) & (y_pred == i))}/{np.sum(class_mask)} correct)")
        else:
            print(f"{class_name}: No test samples")
    
    # Detailed confusion analysis for low-performing classes
    print("\n" + "="*60)
    print("DETAILED CONFUSION ANALYSIS (Low-Performing Classes):")
    print("="*60)
    
    # Find classes with accuracy < 50%
    low_performing_classes = []
    for i in range(num_classes):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.sum((y_true == i) & (y_pred == i)) / np.sum(class_mask)
            if class_accuracy < 0.5:
                low_performing_classes.append((i, class_accuracy))
    
    if low_performing_classes:
        for class_idx, class_acc in low_performing_classes:
            class_name = COMBINED_CLASS_NAMES[class_idx]
            print(f"\n{class_name} (Accuracy: {class_acc * 100:.2f}%):")
            print(f"  Correct predictions: {cm[class_idx][class_idx]}")
            print(f"  Total samples: {np.sum(y_true == class_idx)}")
            print(f"  Misclassified as:")
            
            # Get top misclassifications
            misclassifications = []
            for pred_idx in range(num_classes):
                if pred_idx != class_idx and cm[class_idx][pred_idx] > 0:
                    misclassifications.append((pred_idx, cm[class_idx][pred_idx]))
            
            misclassifications.sort(key=lambda x: x[1], reverse=True)
            for pred_idx, count in misclassifications[:5]:  # Top 5 misclassifications
                pred_name = COMBINED_CLASS_NAMES[pred_idx]
                percentage = (count / np.sum(y_true == class_idx)) * 100
                print(f"    - {pred_name}: {count} ({percentage:.1f}%)")
    else:
        print("\nNo classes with accuracy < 50% found.")
    
    return {
        'accuracy': accuracy,
        'y_true': y_true,
        'y_pred': y_pred,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    # Load model
    print("Loading model...")
    model = load_model(model_path, safe_mode=False, compile=False)
    print("Model loaded successfully!")
    
    # Test data directory
    test_data_dir = '/Users/sajain/Downloads/IMG_CLASSES 2'
    
    # Evaluate model (set num_samples_per_class=None to test on all images, or a number to limit)
    print("\nEvaluating model...")
    results = evaluate_model(model, test_data_dir, num_samples_per_class=100)
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)

