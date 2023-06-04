import io
import base64
from flask import Flask, request, jsonify
import os
import pandas as pd
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from utils.translated_disease import disease_dic
from time import time

# Ajuste das pastas de template e assets

LEAF_FOLDER = os.path.join('template/assets', 'leaf')

app = Flask(__name__, template_folder='template',
            static_folder='template/assets')
app.config['UPLOAD_FOLDER'] = LEAF_FOLDER


# Import do modelo já treinado e salvo (essa parte foi feita no jupyter notebook)
modelo_pipeline = load_model('./models/plant_disease_detection.h5')

disease_list = list(disease_dic.keys())
print(disease_list)


@app.route('/download', methods=['POST'])
def download():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'no image data'}), 400
    img_data = base64.b64decode(data['image'])
    img_json = Image.open(io.BytesIO(img_data))

    milliseconds = int(time() * 1000)
    fullfilename = os.path.join(
        app.config['UPLOAD_FOLDER'], f"image-{milliseconds}.jpg")
    img_json.save(fullfilename)

    img = image.load_img(fullfilename, target_size=(224, 224, 3))

    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255

    prediction = modelo_pipeline.predict(img_tensor)
    print(prediction)
    predicted_class = np.argmax(prediction[0])
    print(predicted_class)

    disease = disease_list[predicted_class]

    return jsonify({'success': 'Image analyzed and saved as ' + fullfilename, 'analise': disease_dic[disease]})


if __name__ == "__main__":
    app.run(debug=True, port=os.getenv("PORT", default=5000))
