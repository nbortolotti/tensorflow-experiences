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


image = tf.keras.preprocessing.image.img_to_array(tf.keras.preprocessing.image.load_img('./pictures/118974357_0faa23cce9_n.jpg', target_size=(128,128))) / 255.


payload = {
    "instances": [{input_name: image.tolist()}]
}
r = requests.post('http://' + host + '/v1/models/' + model_name + ':predict', json=payload)

#print(json.loads(r.content)) # json content format

flowers = np.array(['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips'], dtype=np.object)
info = json.loads(r.content)

for url in info["predictions"]:
    #print np.around(url) array rounded
    print(flowers[np.around(url).astype(np.bool)][0])
