from __future__ import print_function

import argparse
import json
import os.path
import pathlib2 as pathlib
# import pygame
import subprocess
import os
import uuid

import google.oauth2.credentials

from google.assistant.library import Assistant
from google.assistant.library.event import EventType
from google.assistant.library.file_helpers import existing_file
from google.assistant.library.device_helpers import register_device

import picamera
import time
import configparser
import requests
import json
import tensorflow as tf
import numpy as np
from PIL import Image

from object_detection.utils import label_map_util

config = configparser.ConfigParser()
config.read('config.env')
input_name = config.get('model', 'input')
model_name = config.get('model', 'name')
host = config.get('model', 'host')

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError

WARNING_NOT_REGISTERED = """
    This device is not registered. This means you will not be able to use
    Device Actions or see your device in Assistant Settings. In order to
    register this device follow instructions at:

    https://developers.google.com/assistant/sdk/guides/library/python/embed/register-device
"""


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


label_map = label_map_util.load_labelmap('./object_detection/data/mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=90, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def getLabels(boxes, classes, scores, score):
    labels = []
    for i, box in enumerate(np.squeeze(boxes)):
        if (np.squeeze(scores)[i] > score):
            labels.append(category_index[np.squeeze(classes)[i]]['name'])

    return labels


def process_event(event, assistant):
    """Pretty prints events.

    Prints all events that occur with two spaces between each new
    conversation and a single space between turns of a conversation.

    Args:
        event(event.Event): The current event to process.
    """
    global custom_response

    print(event)
    if event.type == EventType.ON_CONVERSATION_TURN_STARTED:
        print()

    if (event.type == EventType.ON_CONVERSATION_TURN_FINISHED and event.args['with_follow_on_turn']):
        print(event.args)

    if (event.type == EventType.ON_CONVERSATION_TURN_FINISHED and
            event.args and not event.args['with_follow_on_turn']):
        print(event.args)
        if custom_response != '':
            assistant.send_text_query('responder burbuja el elefante ha visualizado este bello ' + custom_response)
            custom_response = ''

    if event.type == EventType.ON_DEVICE_ACTION:
        for command, params in event.actions:
            print('Do command', command, 'with params', str(params))

    if event.type == EventType.ON_RENDER_RESPONSE:
        if event.args['text'] in 'Estoy procesando tu imagen, amuleto que dices?':
            # assistant.stop_conversation()
            imgname = ''
            with picamera.PiCamera() as camera:
                imgname = uuid.uuid4().hex.upper()[0:6]
                camera.start_preview()
                time.sleep(2)
                camera.capture('/home/pi/Desktop/' + imgname + '.jpg')
                camera.stop_preview()

            img = Image.open('/home/pi/Desktop/' + imgname + '.jpg')
            image_np = load_image_into_numpy_array(img)

            payload = {
                "instances": [{input_name: image_np.tolist()}]
            }
            r = requests.post('http://' + host + '/v1/models/' + model_name + ':predict', json=payload)

            # print(r.content)

            j = json.loads(r.content.decode('utf-8'))

            # print(j)

            boxes = j['predictions'][0]['detection_boxes']
            classes = j['predictions'][0]['detection_classes']
            scores = j['predictions'][0]['detection_scores']

            detection = getLabels(boxes, classes, scores, 0.3)

            if detection:
                custom_response = detection[0]
            else:
                custom_response = "algo que no puedo reconocer"

def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--device-model-id', '--device_model_id', type=str,
                        metavar='DEVICE_MODEL_ID', required=False,
                        help='the device model ID registered with Google')
    parser.add_argument('--project-id', '--project_id', type=str,
                        metavar='PROJECT_ID', required=False,
                        help='the project ID used to register this device')
    parser.add_argument('--device-config', type=str,
                        metavar='DEVICE_CONFIG_FILE',
                        default=os.path.join(
                            os.path.expanduser('~/.config'),
                            'googlesamples-assistant',
                            'device_config_library.json'
                        ),
                        help='path to store and read device configuration')
    parser.add_argument('--credentials', type=existing_file,
                        metavar='OAUTH2_CREDENTIALS_FILE',
                        default=os.path.join(
                            os.path.expanduser('~/.config'),
                            'google-oauthlib-tool',
                            'credentials.json'
                        ),
                        help='path to store and read OAuth2 credentials')
    parser.add_argument('-v', '--version', action='version',
                        version='%(prog)s ' + Assistant.__version_str__())

    args = parser.parse_args()
    with open(args.credentials, 'r') as f:
        credentials = google.oauth2.credentials.Credentials(token=None,
                                                            **json.load(f))

    device_model_id = None
    last_device_id = None
    try:
        with open(args.device_config) as f:
            device_config = json.load(f)
            device_model_id = device_config['model_id']
            last_device_id = device_config.get('last_device_id', None)
    except FileNotFoundError:
        pass

    if not args.device_model_id and not device_model_id:
        raise Exception('Missing --device-model-id option')

    # Re-register if "device_model_id" is given by the user and it differs
    # from what we previously registered with.
    should_register = (
            args.device_model_id and args.device_model_id != device_model_id)

    device_model_id = args.device_model_id or device_model_id

    with Assistant(credentials, device_model_id) as assistant:
        events = assistant.start()

        device_id = assistant.device_id
        print('device_model_id:', device_model_id)
        print('device_id:', device_id + '\n')

        # Re-register if "device_id" is different from the last "device_id":
        if should_register or (device_id != last_device_id):
            if args.project_id:
                register_device(args.project_id, credentials,
                                device_model_id, device_id)
                pathlib.Path(os.path.dirname(args.device_config)).mkdir(
                    exist_ok=True)
                with open(args.device_config, 'w') as f:
                    json.dump({
                        'last_device_id': device_id,
                        'model_id': device_model_id,
                    }, f)
            else:
                print(WARNING_NOT_REGISTERED)

        for event in events:
            process_event(event, assistant)


if __name__ == '__main__':
    main()
