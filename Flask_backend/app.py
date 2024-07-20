from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from flask_cors import CORS
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load your model
model = load_model('models/mymodel.keras')

# Ensure 'uploads' directory exists
if not os.path.exists('uploads'):
    os.makedirs('uploads')

# Define the classes
classes = [
    'actinic keratosis', 'basal cell carcinoma', 'dermatofibroma',
    'melanoma', 'nevus', 'pigmented benign keratosis',
    'seborrheic keratosis', 'squamous cell carcinoma', 'vascular lesion'
]

def model_predict(img_path, model):
    img = load_img(img_path, target_size=(128, 128))  # Adjust target size based on your model
    img_array = img_to_array(img) / 255.0  # Normalize image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    predictions = model.predict(img_array)
    return predictions

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Save the file to the 'uploads' directory
        file_path = os.path.join('uploads', secure_filename(file.filename))
        file.save(file_path)
        
        # Perform prediction using the saved file
        predictions = model_predict(file_path, model)
        max_index = np.argmax(predictions[0])
        confidence = float(predictions[0][max_index])
        
        predicted_class = classes[max_index]
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)