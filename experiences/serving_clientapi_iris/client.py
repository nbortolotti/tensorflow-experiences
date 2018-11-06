import ConfigParser
import requests
import json
import numpy as np

config = ConfigParser.ConfigParser()
config.read('config.env')

input_name = config.get('model', 'input')
model_name = config.get('model', 'name')
host = config.get('model', 'host')

ar = np.array([7.9, 3.8, 6.4, 2.0]) # example to test iris model

payload = {
    "instances": [{input_name: ar.tolist()}]
}
r = requests.post('http://' + host + '/v1/models/' + model_name + ':predict', json=payload)
print(json.loads(r.content))
