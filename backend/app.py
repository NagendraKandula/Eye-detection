from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tensorflow as tf
import imghdr

app = Flask(__name__)
CORS(app)

# Load the retina check model (binary classifier)
retina_interpreter = tf.lite.Interpreter(model_path="model_binaray.tflite")
retina_interpreter.allocate_tensors()
retina_input_details = retina_interpreter.get_input_details()
retina_output_details = retina_interpreter.get_output_details()

# Load the DenseNet disease classifier model
disease_interpreter = tf.lite.Interpreter(model_path="densenet.tflite")
disease_interpreter.allocate_tensors()
disease_input_details = disease_interpreter.get_input_details()
disease_output_details = disease_interpreter.get_output_details()

class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']
image_size = (224, 224)

def preprocess_image_for_tflite(img, input_details):
    img = img.resize(image_size).convert('RGB')
    img_array = np.array(img)

    # Handle input type (e.g., float32 or uint8)
    input_type = input_details[0]['dtype']
    if input_type == np.uint8:
        img_array = np.expand_dims(img_array, axis=0).astype(np.uint8)
    else:  # default to float32
        img_array = np.expand_dims(img_array / 255.0, axis=0).astype(np.float32)

    return img_array

def is_retina_image(img):
    img_array = preprocess_image_for_tflite(img, retina_input_details)
    retina_interpreter.set_tensor(retina_input_details[0]['index'], img_array)
    retina_interpreter.invoke()
    output_data = retina_interpreter.get_tensor(retina_output_details[0]['index'])
    prediction = np.round(output_data[0][0])
    return bool(prediction)

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

        # Step 1: Retina image validation
        if not is_retina_image(img):
            return jsonify({'error': 'Uploaded image is not a retina image. Please upload a valid retina image.'}), 400

        # Step 2: Disease classification
        img_array = preprocess_image_for_tflite(img, disease_input_details)
        disease_interpreter.set_tensor(disease_input_details[0]['index'], img_array)
        disease_interpreter.invoke()
        predictions = disease_interpreter.get_tensor(disease_output_details[0]['index'])

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
    app.run(debug=True)
