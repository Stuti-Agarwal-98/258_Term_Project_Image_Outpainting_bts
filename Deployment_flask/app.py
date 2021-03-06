import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect
# from werkzeug.utils import secure_filename
# from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import (ImageDataGenerator, array_to_img, img_to_array, load_img)

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model

import time
# Some utilites
import numpy as np
#from util import base64_to_pil
import PIL
# Declare a flask app
app = Flask(__name__)


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# or https://www.tensorflow.org/api_docs/python/tf/keras/applications


outpainting_model = keras.models.load_model("./models/")

#plant_patho_labels = ["healthy", "multiple_diseases", "rust", "scab"]

print('Models loaded. Check http://127.0.0.1:5000/')
#bts
IMAGE_SZ = 128
padding_width = int(IMAGE_SZ / 4)

def get_img_path():
    gallery_images = glob.glob('./input_images/*')
    print(gallery_images)
    return gallery_images,render_template('index.html', filename=galery_images)
    
def get_masked_images(imgs, padding_width):
    padded_imgs = np.copy(imgs)
    pix_avg = np.mean(padded_imgs, axis=(1, 2, 3))
    print(pix_avg.shape)
    padded_imgs[:, :, :padding_width, :] = padded_imgs[:, :, -padding_width:, :] = pix_avg.reshape(-1, 1, 1, 1) #calculating mean pixel intensity and place it masked pixels
    padded_imgs[:, :padding_width, :, :] = padded_imgs[:, -padding_width:, :, :] = pix_avg.reshape(-1, 1, 1, 1) #calculating mean pixel intensty and place it masked pixelsi

    return padded_imgs
def crop_and_resize_image(img, img_size):
    source_size = img.size
    if source_size == img_size:
        return img
    img = img.resize(img_size)
    return img
    
def renorm_image(img_norm):
    img_renorm = (img_norm *255).astype(np.uint8)
    return img_renorm
    
def image_transform(image):
    image_array = []
    img_size = (128,128)   
    #f = Image.open(r"C:\Users\stuti\258_flask\Deployment-flask\input_images\Places365_val_00000102.jpg") 
    img = crop_and_resize_image(image, img_size)
    numpy_img = img_to_array(img)
    image_array.append(numpy_img)
    imgs = np.asarray(image_array, dtype=np.float32)/255.0

    masked_image = get_masked_images(imgs,padding_width)
    return masked_image
    

@app.route('/')
def home():
    #get_img_path()
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        
        # Get the image from post request
        f = keras.preprocessing.image.load_img("./static/input_images/Places365_val_00000012.jpg")
        outpainting_input_img = image_transform(f)
        masked_image = renorm_image(outpainting_input_img)
        input_image = PIL.Image.fromarray(masked_image[0],'RGB')
        input_image.save("./input/output_pred.jpg")
        print("Transformed Image")
        print(type(outpainting_model))
        outpainting_preds = outpainting_model.predict(outpainting_input_img)
        new_img = renorm_image(outpainting_preds)
        final_pred_img = PIL.Image.fromarray(new_img[0],'RGB')

        # new_size = (256,256)
        # final_pred_img = final_pred_img.resize(new_size)
        time.sleep(4)
        final_pred_img.save("./static/output/output_pred.jpg")
        return render_template('index.html', output_filename='./output/output_pred.jpg')

# def test():

#     f = keras.preprocessing.image.load_img("./static/input_images/Places365_val_00000012.jpg")
#     outpainting_input_img = image_transform(f)
#     masked_image = renorm_image(outpainting_input_img)
#     input_image = PIL.Image.fromarray(masked_image[0],'RGB')
#     input_image.save("./input/output_pred.jpg")
#     print("Transformed Image")
#     print(type(outpainting_model))
#     outpainting_preds = outpainting_model.predict(outpainting_input_img)
#     new_img = renorm_image(outpainting_preds)
#     final_pred_img = PIL.Image.fromarray(new_img[0],'RGB')
#     final_pred_img.save("./output/output_pred.jpg")
    
# test()
if __name__ == "__main__":
    app.run(debug=True)