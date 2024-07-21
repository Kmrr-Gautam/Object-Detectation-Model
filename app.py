from flask import Flask, request, render_template, redirect, url_for
import os
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    return image

def decode_predictions(preds):
    return tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=3)[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            with open(file_path, 'rb') as f:
                image = Image.open(f)
                image = image.convert('RGB')

            processed_image = preprocess_image(image)
            preds = model.predict(processed_image)
            decoded_preds = decode_predictions(preds)

            top_pred = decoded_preds[0]
            label = top_pred[1]
            score = top_pred[2]

            return render_template('result.html', filename=file.filename, label=label, score=score)

    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
