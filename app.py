import os, json
import cv2 as cv
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

class CVPipeLine(object):
    def __init__(self):

        self.classification_input_shape = (224, 224, 3)

        classification_weights_corn = 'weights/disease classification - corn.h5'
        classification_weights_apple = 'weights/disease classification - apple.h5'
        classification_weights_potato = 'weights/disease classification - potato.h5'
        classification_weights_bell_pepper = 'weights/disease classification - bell_pepper.h5'

        self.corn_disease_classifier = tf.keras.models.load_model(classification_weights_corn)
        self.corn_disease_classifier.compile(
                                                optimizer='Adam',
                                                loss='categorical_crossentropy',
                                                metrics = [
                                                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                            tf.keras.metrics.Precision(name='precision'),
                                                            tf.keras.metrics.Recall(name='recall')
                                                            ]
                                                )

        self.apple_disease_classifier = tf.keras.models.load_model(classification_weights_apple)
        self.apple_disease_classifier.compile(                                            
                                                optimizer='Adam',
                                                loss='categorical_crossentropy',
                                                metrics = [
                                                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                            tf.keras.metrics.Precision(name='precision'),
                                                            tf.keras.metrics.Recall(name='recall')
                                                            ]
                                                )

        self.potato_disease_classifier = tf.keras.models.load_model(classification_weights_potato)
        self.potato_disease_classifier.compile(                                            
                                                optimizer='Adam',
                                                loss='categorical_crossentropy',
                                                metrics = [
                                                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                            tf.keras.metrics.Precision(name='precision'),
                                                            tf.keras.metrics.Recall(name='recall')
                                                            ]
                                                )

        self.bell_pepper_disease_classifier = tf.keras.models.load_model(classification_weights_bell_pepper)
        self.bell_pepper_disease_classifier.compile(                                            
                                                optimizer='Adam',
                                                loss='categorical_crossentropy',
                                                metrics = [
                                                            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
                                                            tf.keras.metrics.Precision(name='precision'),
                                                            tf.keras.metrics.Recall(name='recall')
                                                            ]
                                                )



        self.segmentation_input_shape = (256, 256, 3)
        self.segmentation_output_shape_orig = (256, 256, 2)
        self.segmentation_output_shape_new = (65536, 2)
        
        segmentation_weights = 'weights/disease segmentation.h5'
        self.disease_segmentation_model = tf.keras.models.load_model(segmentation_weights)

        recall = tf.keras.metrics.Recall()
        precision = tf.keras.metrics.Precision()

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                                                                    0.0005,
                                                                    decay_steps=100000,
                                                                    decay_rate=0.96,
                                                                    staircase=True
                                                                    )

        self.disease_segmentation_model.compile(
                                            loss='categorical_crossentropy',
                                            optimizer=tf.keras.optimizers.Adam(lr_schedule),
                                            metrics=['accuracy', recall, precision],
                                            sample_weight_mode="temporal"
                                            )

    def process_classification_image(self, image_file_path):
        image = cv.imread(image_file_path)
        image = cv.resize(image, self.classification_input_shape[0:2], interpolation = cv.INTER_AREA)
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
        image = np.expand_dims(image, axis=0)
        return image

    def classification_inference(self, image_file_path, plant):
        image_pp = self.process_classification_image(image_file_path)
        if plant == 'corn':
            prediction = self.corn_disease_classifier.predict(image_pp).squeeze()
            prediction = np.argmax(prediction)
            class_dict = {
                        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 0,
                        'Corn_(maize)___Common_rust_': 1,
                        'Corn_(maize)___Northern_Leaf_Blight': 2,
                        'Corn_(maize)___healthy': 3
                        }
            class_dict_rev = {v: k for k, v in class_dict.items()}

        elif plant == 'apple':
            prediction = self.apple_disease_classifier.predict(image_pp).squeeze()
            prediction = np.argmax(prediction)
            class_dict = {
                        'Apple___Apple_scab': 0,
                        'Apple___Black_rot': 1,
                        'Apple___Cedar_apple_rust': 2,
                        'Apple___healthy': 3
                         }
            class_dict_rev = {v: k for k, v in class_dict.items()}

        elif plant == 'potato':
            prediction = self.potato_disease_classifier.predict(image_pp).squeeze()
            prediction = np.argmax(prediction)
            class_dict = {
                        'Potato___Early_blight': 0, 
                        'Potato___Late_blight': 1, 
                        'Potato___healthy': 2
                        }
            class_dict_rev = {v: k for k, v in class_dict.items()}

        elif plant == 'bell_pepper':
            prediction = self.bell_pepper_disease_classifier.predict(image_pp).squeeze()
            prediction = np.argmax(prediction)
            class_dict = {
                        'Pepper,_bell___Bacterial_spot': 0, 
                        'Pepper,_bell___healthy': 1
                        }
            class_dict_rev = {v: k for k, v in class_dict.items()}

        else:
            raise ValueError('Invalid plant name')

        return class_dict_rev[prediction]
        
    def preprocess_segmentation_image(self, image_file_path):
        img = cv.imread(image_file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = np.asarray(img).astype("f")
        img = cv.resize(
                      img, 
                      (self.segmentation_input_shape[0], self.segmentation_input_shape[1]), 
                      interpolation = cv.INTER_AREA
                       )
        img = np.expand_dims(img, axis=0)
        return img/255.0

    def segmentation_inference(self, image_file_path):
        img = self.preprocess_segmentation_image(image_file_path)
        mask = self.disease_segmentation_model.predict(img)
        mask = mask.argmax(axis=-1)
        flatten_mask = np.squeeze(mask)

        total_white_fixcels = np.count_nonzero(flatten_mask == 1)
        total_black_fixcels = np.count_nonzero(flatten_mask == 0)

        disease_pixel_percentage = (total_white_fixcels / (total_white_fixcels + total_black_fixcels)) * 100
        disease_pixel_percentage = round(disease_pixel_percentage, 2)
        disease_pixel_percentage = f"{disease_pixel_percentage} %"
        return disease_pixel_percentage

    def disease_image_integration(self, image_file_path, plant):
        disease = self.classification_inference(image_file_path, plant)
        disease_pixel_percentage = self.segmentation_inference(image_file_path)
        return disease, disease_pixel_percentage

P = CVPipeLine()

@app.route('/predict', methods=['POST'])
def predict():
    plant = str(request.form['plant'])
    image = request.files['image']
    image_file_path = os.path.join('upload', secure_filename(image.filename))
    image.save(image_file_path)

    disease, disease_pixel_percentage = P.disease_image_integration(image_file_path, plant)
    response =  {
            "disease": f"{disease}",
            "disease_propogation": f"{disease_pixel_percentage}"
                }
    return json.dumps(response)

if __name__ == '__main__':
    app.run(
            host='0.0.0.0',
            port=5000,
            debug=True
            )