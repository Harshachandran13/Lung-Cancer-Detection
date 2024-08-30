import os
import tensorflow as tf
import numpy as np
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from keras.preprocessing import image

# Define a flask app
app = Flask(__name__, template_folder='template')

# Load the model
model = tf.keras.models.load_model('Lungs.h5', compile=False)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = np.array(x, 'float32') / 255
    preds = model.predict(x)
    return preds

@app.route("/", methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def upload():
    if request.method == 'POST' and 'file' in request.files:
        # Get the file from the post request
        f = request.files['file']
        # Save the file to ./upload
        file_path = os.path.join('upload', secure_filename(f.filename))
        f.save(file_path)
        # Make prediction
        preds = model_predict(file_path, model)
        disease_class = ['Benign', 'Lung Cancer Detected']
        ind = np.argmax(preds[0])
        result = disease_class[ind]
        return result
    return "No file uploaded"

if __name__ == '__main__':
    app.run(debug=True)
