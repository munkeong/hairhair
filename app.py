import os
from PIL import Image
from flask import Flask, request, Response
import tensorflow as tf
import numpy as np
import cv2
import io
from tflite_model import *
import json
app = Flask(__name__)

#init tflite model
model_hair = Model("hair_segmentation.tflite")
in_shape = model_hair.getInputShape()
h = in_shape[1]
w = in_shape[2]

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@app.route('/')
def index():
    return Response('MECa3 Object Detection Test 2019.09.27 #8')


@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")


@app.route('/video')
def remote():
    return Response(open('./static/video.html').read(), mimetype="text/html")


@app.route('/test')
def test():
    PATH_TO_TEST_IMAGES_DIR = 'object_detection/test_images'  # cwh
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]
    threshold = request.form.get('threshold')
    if threshold is None:
        threshold = 2.5
    img = cv2.imread(TEST_IMAGE_PATHS)
    img_reszd = cv2.resize(img, (w, h))
    img_pre = (img_reszd / 255 - 0.5) * 2
    output_tensors = np.squeeze(model_hair.runModel(img_pre))
    output_json = hair_json(output_tensors, threshold)
    return output_json

@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image'].read()  # get the image

        # Set an image confidence threshold value to limit returned data
        #threshold = request.form.get('threshold')

        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 2.5
        else:
            threshold = float(threshold)

        img = np.array(Image.open(io.BytesIO(image_file)))

	#img = cv2.imread(image_file)
        img_reszd = cv2.resize(img, (w, h))
        img_pre = ((img_reszd / 255 - 0.5) * 2).astype('float32')

        in_tensor = np.zeros(in_shape[1:],dtype=np.float32)
        in_tensor[:,:,0:3] = img_pre

        output_tensors = np.squeeze(model_hair.runModel(in_tensor))
        output_json = hair_json(output_tensors, threshold) #threshold)
        return output_json

    except Exception as e:
        print('POST /image error: %e' % e)
        return e

if __name__ == '__main__':
	# without SSL
    app.run(debug=True, host='0.0.0.0')
    #app.run(debug=True)

	# with SSL
    #app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
