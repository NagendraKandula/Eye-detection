from flask import Flask, request, jsonify
from flask_cors import CORS  
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)  

# Load the trained model once
model = load_model('eye_disease_model.tflite')

# Define class names
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
image_size = (224, 224)  
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file received'}), 400

    try:
        # Preprocess the uploaded image
        img = Image.open(file).convert('RGB').resize(image_size)
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        predictions = model.predict(img_array)
        class_index = int(np.argmax(predictions[0]))
        accuracy = float(np.max(predictions[0])) * 100.0

        result = {
            'model': 'Eye Disease Classifier ',
            'name': class_names[class_index],
            'predicted_class': class_index,
            'accuracy': f"{accuracy:.2f}%",
            'remedy': suggest_remedy(class_names[class_index])
        }

        return jsonify({'result': [result]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def suggest_remedy(class_name):
    remedies = {
        'cataract': 'Consult an ophthalmologist for surgery options.',
        'diabetic_retinopathy': 'Maintain blood sugar levels and get regular eye checkups.',
        'glaucoma': 'Use prescribed eye drops and monitor eye pressure regularly.',
        'normal': 'Your eyes appear normal. Maintain a healthy lifestyle.'
    }
    return remedies.get(class_name, 'No specific remedy found.')

if __name__ == '__main__':
    app.run(debug=True)
