from flask import Flask, request, jsonify
from flask_cors import CORS  
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import imghdr
import os  # Required for os.environ

app = Flask(__name__)
CORS(app)  

# Load the trained DenseNet model once
model = load_model('densenet.keras')

# Load TFLite retina check model once
interpreter = tf.lite.Interpreter(model_path="model_binaray.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
image_size = (224, 224)

def preprocess_image_for_tflite(img):
    img = img.resize(image_size).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
    return img_array

def is_retina_image(img):
    img_array = preprocess_image_for_tflite(img)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.round(output_data[0][0])
    return bool(prediction)  # True if retina image, False otherwise

def preprocess_image_for_classification(img):
    img = img.resize(image_size).convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty file received'}), 400

    try:
        file_bytes = file.read()
        file_type = imghdr.what(None, h=file_bytes)
        if file_type not in ['jpeg', 'png']:
            return jsonify({'error': 'Invalid image format. Use JPEG or PNG.'}), 400

        file.stream.seek(0)
        img = Image.open(file).convert('RGB')

        # Step 1: Retina check
        if not is_retina_image(img):
            return jsonify({'error': 'Uploaded image is not a retina image. Please upload a valid retina image.'}), 400

        # Step 2: Eye disease classification
        img_array = preprocess_image_for_classification(img)
        predictions = model.predict(img_array)

        if predictions.shape[-1] != len(class_names):
            return jsonify({'error': 'Model output shape mismatch.'}), 500

        class_index = int(np.argmax(predictions[0]))
        accuracy = float(np.max(predictions[0])) * 100.0

        result = {
            'model': 'Eye Disease Classifier',
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
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))  # Fixed space
