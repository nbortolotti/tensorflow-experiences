import picamera
from flask import Flask

import time
import ConfigParser
import requests
import json
import tensorflow as tf
import numpy as np

config = ConfigParser.ConfigParser()
config.read('config.env')

input_name = config.get('model', 'input')
model_name = config.get('model', 'name')
host = config.get('model', 'host')

app = Flask(__name__)
app._static_folder = "/home/pi/Desktop/servingflask"

import uuid;


@app.route("/picture")
def takepicture():
    with picamera.PiCamera() as camera:
        imgname = uuid.uuid4().hex.upper()[0:6]
        camera.start_preview()
        time.sleep(5)
        camera.capture(imgname + '.jpg')
        camera.stop_preview()

        app.send_static_file(imgname + '.jpg')

        time.sleep(7)
        image = tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img('/home/pi/Desktop/servingflask/' + imgname + '.jpg',
                                                  target_size=(128, 128))) / 255.

        payload = {
            "instances": [{input_name: image.tolist()}]
        }
        r = requests.post('http://' + host + '/v1/models/' + model_name + ':predict', json=payload)

        flowers = np.array(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'], dtype=np.object)
        info = json.loads(r.content)

        # return app.response_class(info, content_type='application/json')
        for url in info["predictions"]:
            # print np.around(url) array rounded
            # print(flowers[np.around(url).astype(np.bool)][0])
            return flowers[np.around(url).astype(np.bool)][0]
    # return "success"


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80, debug=True)