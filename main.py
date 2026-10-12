import io
import os

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model

# loading class names (same order as the training folders)
CLASS_NAMES = ['Grad de risc scăzut', 'Grad de risc crescut', 'Piele curată']

app = Flask(__name__)
# reject uploads larger than 20 MB
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024

# loading the model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model2.h5')
model = load_model(MODEL_PATH, compile=False)

# adjusting input size to match the model
IMG_SIZE = (180, 180)


def preprocess_image(image_bytes):
    # Must match training (cnn.ipynb): image_dataset_from_directory resizes
    # RGB images bilinearly to 180x180 and keeps raw 0-255 pixel values,
    # because the model starts with its own Rescaling(1/255) layer.
    # No extra contrast enhancement is applied here, since none was applied
    # during training.
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(IMG_SIZE, Image.BILINEAR)

    image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_bytes = request.files['image'].read()
    try:
        img_array = preprocess_image(image_bytes)  # shape (1, 180, 180, 3)
    except UnidentifiedImageError:
        return jsonify({'error': 'Invalid image file'}), 400

    try:
        preds = model.predict(img_array, verbose=0)
        class_id = int(np.argmax(preds[0]))
        confidence = float(np.max(preds[0]))
        result = {
            'predicted_class_id': class_id,
            'predicted_class_name': CLASS_NAMES[class_id],
            'confidence': confidence
        }
        return jsonify(result)
    except Exception as e:
        print(e)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # the reloader would load the 170 MB model twice, so keep it off
    app.run(host='0.0.0.0', port=port, debug=False)
