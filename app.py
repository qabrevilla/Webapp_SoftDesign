from flask import Flask, render_template, request, redirect, url_for, flash
import os
import numpy as np
from werkzeug.utils import secure_filename
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import librosa
import librosa.display
from skimage.transform import resize

app = Flask(__name__)
app.secret_key = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def create_vgg_model(num_classes):
    vgg_model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)  # pre-trained
    vgg_model.classifier[6] = nn.Linear(vgg_model.classifier[6].in_features, num_classes)

    # Freeze convolutional layers 
    for param in vgg_model.features.parameters():
        param.requires_grad = False
        
    return vgg_model

# Global counters (initialized when the app starts)
defect_counters = {
    'Crack': 0,
    'Scratch': 0,
    'Stain': 0,
    'Sound Defect': 0
}

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

visual_model = create_vgg_model(4)
visual_model.to(device)
visual_model.load_state_dict(torch.load(r"models\visual_vgg_weights.pt", weights_only=True))

sound_model = create_vgg_model(2)
sound_model.to(device)
sound_model.load_state_dict(torch.load(r"models\sound_vgg_weights2.pt", weights_only=True))


# Helper Functions
def preprocess_image(image_path):   # VISUAL
    preprocess = transforms.Compose([
    transforms.Resize((299, 299)),  # Resize images to 224x224 (compatible with ResNet)
    transforms.ToTensor(),          # Convert PIL images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained ResNet
    ])
    image = Image.open(image_path).convert('RGB')
    preprocessed_image = preprocess(image)
    return preprocessed_image

def preprocess_sound(sound_path, target_size=(224, 224)):   # SOUND
    # Load audio
    signal, sample_rate = librosa.load(sound_path, sr=None)
    
    # Generate spectrogram
    stft = librosa.stft(signal, n_fft=2048, hop_length=512)
    spectrogram = np.abs(stft)
    
    # Convert to log scale
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    
    # Resize to fixed size
    resized_spectrogram = resize(log_spectrogram, target_size, mode='constant')
    
    return resized_spectrogram

def process_sound(spectrogram, model, class_names=['Good', 'Bad'], device='cuda'):
    """
    Predicts whether a spectrogram indicates a defect or not using a PyTorch model.

    Parameters:
        spectrogram (numpy.ndarray): The preprocessed spectrogram array.
        model (torch.nn.Module): Trained VGG16 PyTorch model for classification.
        class_names (list): List of class names. Default is ['Good', 'Bad'].
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        dict: A dictionary containing the predicted class and the confidence score.
    """
    try:
        preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),          # Convert PIL images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained ResNet
        ])
        
        # Ensure spectrogram has the correct dimensions
        # if len(spectrogram.shape) != 2:
        #     raise ValueError("Spectrogram must be a 2D array.")

        # Normalize the spectrogram (scale between 0 and 1)
        spectrogram = spectrogram / np.max(spectrogram)

        # Convert the spectrogram to a PyTorch tensor
        spectrogram_tensor = torch.tensor(spectrogram, dtype=torch.float32)

        # Add batch and channel dimensions
        spectrogram_tensor = spectrogram_tensor.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, H, W)

        # Repeat the channel dimension to make it compatible with VGG16 (3 channels)
        spectrogram_tensor = spectrogram_tensor.repeat(1, 3, 1, 1)  # Shape: (1, 3, H, W)

        # Resize the spectrogram to fit VGG16's input (224x224)
        spectrogram_tensor = F.interpolate(spectrogram_tensor, size=(224, 224), mode='bilinear')

        # Move tensor to the specified device
        spectrogram_tensor = spectrogram_tensor.to(device)

        # Set the model to evaluation mode
        model.eval()

        # Perform inference
        with torch.no_grad():
            output = model(spectrogram_tensor)

        # Compute probabilities using softmax
        probabilities = torch.softmax(output, dim=1).squeeze()

        # Get the predicted class and confidence
        predicted_class_idx = torch.argmax(probabilities).item()
        confidence = probabilities[predicted_class_idx].item()

        predicted_class = class_names[predicted_class_idx]
        
        if predicted_class == 'Bad':
            defect_counters['Sound Defect'] += 1
        
        return {
            "Sound Quality": 1 if predicted_class == 'Bad' else 0,
            "Confidence": confidence
        }
    except Exception as e:
        return {"error": str(e)}

def process_visual(model, preprocessed_input, device):
    """
    Processes the visual model's prediction.
    """
    model.eval()
    with torch.no_grad():
        preprocessed_input = preprocessed_input.to(device)
        output = model(preprocessed_input.unsqueeze(0))
        sigmoid_output = torch.sigmoid(output).squeeze()
        binary_output = (sigmoid_output > 0.5).int().cpu().numpy()

        # Define label mapping
        labels = ['Crack', 'Normal', 'Scratch', 'Stain']

        # Map binary output to defect labels
        visual_results = {labels[idx]: int(binary_output[idx]) for idx in range(len(labels))}

        # Update counters
        for label, detected in visual_results.items():
            if detected == 1:
                defect_counters[label] += 1
        
        return visual_results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        image_file = request.files.get('image')
        sound_file = request.files.get('sound')
        if not image_file or not sound_file:
            flash("Both image and sound files are required.")
            return redirect(url_for('index'))

        image_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
        sound_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(sound_file.filename))
        image_file.save(image_path)
        sound_file.save(sound_path)

        # Process inputs
        p_image = preprocess_image(image_path)
        p_sound = preprocess_sound(sound_path)
        # Get predictions
        visual_results = process_visual(visual_model, p_image, device)
        sound_results = process_sound(p_sound, sound_model)

        # Combine results
        inspection_summary = {
            'Crack': visual_results['Crack'],
            'Scratch': visual_results['Scratch'],
            'Stain': visual_results['Stain'],
            'Sound Defect': sound_results['Sound Quality'],
        }

        return render_template('index.html',
                               visual_results=visual_results,
                               sound_results=sound_results,
                               inspection_summary=inspection_summary,
                               defect_counters=defect_counters
                               )

    return render_template('index.html', defect_counters=defect_counters)

if __name__ == '__main__':
    app.run(debug=True)
