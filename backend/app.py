import logging
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import imghdr

app = Flask(__name__)
CORS(app)

# Enable logging
logging.basicConfig(level=logging.INFO)

# Globals
retina_interpreter = None
retina_input_details = None
retina_output_details = None

disease_interpreter = None
disease_input_details = None
disease_output_details = None

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
image_size = (224, 224)

def get_retina_model():
    global retina_interpreter, retina_input_details, retina_output_details
    if retina_interpreter is None:
        logging.info("Loading retina model")
        retina_interpreter = tf.lite.Interpreter(model_path="model_binaray.tflite")
        retina_interpreter.allocate_tensors()
        retina_input_details = retina_interpreter.get_input_details()
        retina_output_details = retina_interpreter.get_output_details()
    return retina_interpreter, retina_input_details, retina_output_details

def get_disease_model():
    global disease_interpreter, disease_input_details, disease_output_details
    if disease_interpreter is None:
        logging.info("Loading disease model")
        disease_interpreter = tf.lite.Interpreter(model_path="densenet.tflite")
        disease_interpreter.allocate_tensors()
        disease_input_details = disease_interpreter.get_input_details()
        disease_output_details = disease_interpreter.get_output_details()
    return disease_interpreter, disease_input_details, disease_output_details

def preprocess_image_for_tflite(img, input_details):
    img = img.resize(image_size).convert('RGB')
    img_array = np.array(img)
    input_type = input_details[0]['dtype']
    if input_type == np.uint8:
        img_array = np.expand_dims(img_array, axis=0).astype(np.uint8)
    else:
        img_array = np.expand_dims(img_array / 255.0, axis=0).astype(np.float32)
    return img_array

def is_retina_image(img):
    retina_interpreter, retina_input_details, retina_output_details = get_retina_model()
    img_array = preprocess_image_for_tflite(img, retina_input_details)
    retina_interpreter.set_tensor(retina_input_details[0]['index'], img_array)
    retina_interpreter.invoke()
    output_data = retina_interpreter.get_tensor(retina_output_details[0]['index'])
    prediction = np.round(output_data[0][0])
    logging.info(f"Retina validation result: {prediction}")
    return bool(prediction)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            logging.warning("No file part in request")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            logging.warning("Empty file received")
            return jsonify({'error': 'Empty file received'}), 400

        file_bytes = file.read()
        file_type = imghdr.what(None, h=file_bytes)
        if file_type not in ['jpeg', 'png']:
            logging.warning("Invalid file format")
            return jsonify({'error': 'Invalid image format. Use JPEG or PNG.'}), 400

        file.stream.seek(0)
        img = Image.open(file).convert('RGB')

        if not is_retina_image(img):
            logging.info("Image failed retina validation")
            return jsonify({'error': 'Uploaded image is not a retina image. Please upload a valid retina image.'}), 400

        disease_interpreter, disease_input_details, disease_output_details = get_disease_model()
        img_array = preprocess_image_for_tflite(img, disease_input_details)
        disease_interpreter.set_tensor(disease_input_details[0]['index'], img_array)
        disease_interpreter.invoke()
        predictions = disease_interpreter.get_tensor(disease_output_details[0]['index'])

        logging.info(f"Model raw predictions: {predictions}")

        if predictions.shape[-1] != len(class_names):
            logging.error("Prediction shape mismatch")
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

        logging.info(f"Final result: {result}")
        return jsonify({'result': [result]})

    except Exception as e:
        logging.error("Exception during prediction:")
        traceback.print_exc()
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
