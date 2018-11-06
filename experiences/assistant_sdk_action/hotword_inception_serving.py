from __future__ import print_function

import argparse
import os.path
import pathlib2 as pathlib
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

config = configparser.ConfigParser()
config.read('config.env')
input_name = config.get('model', 'input')
model_name = config.get('model', 'name')
host = config.get('model', 'host')

from google.cloud import texttospeech

# Instantiates a client
client = texttospeech.TextToSpeechClient()

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

local_commands = ['retrato', 'captura', 'imagen']


def process_event(event, assistant):
    """Pretty prints events.

    Prints all events that occur with two spaces between each new
    conversation and a single space between turns of a conversation.

    Args:
        event(event.Event): The current event to process.
    """
    if event.type == EventType.ON_CONVERSATION_TURN_STARTED:
        print()
    # print('atencion')
    print(event)
    # print('atecion fin')

    if (event.type == EventType.ON_CONVERSATION_TURN_FINISHED and
            event.args and not event.args['with_follow_on_turn']):
        print()
    if event.type == EventType.ON_DEVICE_ACTION:
        for command, params in event.actions:
            print('Do command', command, 'with params', str(params))

    if event.type == EventType.ON_DEVICE_ACTION:
        for command, params in event.actions:
            print('Do command', command, 'with params', str(params))

            # Add the following lines after the existing line above:

            if command == "com.example.actions.fotografia":
                print('comando')
    # if event.type == EventType.ON_RECOGNIZING_SPEECH_FINISHED:
    #    print(event.args['text'])
    #    if event.args['text'] in local_commands:
    if event.type == EventType.ON_RENDER_RESPONSE:
        if event.args['text'] in 'Estoy procesando tu imagen, amuleto que dices?':
            assistant.stop_conversation()
            print('exito')

            imgname = ''
            with picamera.PiCamera() as camera:
                imgname = uuid.uuid4().hex.upper()[0:6]
                camera.start_preview()
                time.sleep(5)
                camera.capture('/home/pi/Desktop/' + imgname + '.jpg')
                camera.stop_preview()

            image = tf.keras.preprocessing.image.img_to_array(
                tf.keras.preprocessing.image.load_img('/home/pi/Desktop/' + imgname + '.jpg',
                                                      target_size=(128, 128))) / 255.

            payload = {
                "instances": [{input_name: image.tolist()}]
            }
            r = requests.post('http://' + host + '/v1/models/' + model_name + ':predict', json=payload)
            print(r)

            flowers = np.array(['margarita', 'diente de leon', 'rosa', 'girasol', 'tulipan'], dtype=np.object)
            try:
                info = json.loads(r.content.decode('utf-8'))
                print(r.content.decode('utf-8'))
                for url in info["predictions"]:
                    cap = (np.array(url) > 0.6).astype(np.bool)
                    print(cap)
                    cap2 = flowers[cap][0]

                    synthesis_input = texttospeech.types.SynthesisInput(
                        text="Estoy contemplando una muy linda flor, es mas problable que sea " + cap2)

                    voice = texttospeech.types.VoiceSelectionParams(
                        language_code='es-ES',
                        ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

                    audio_config = texttospeech.types.AudioConfig(
                        audio_encoding=texttospeech.enums.AudioEncoding.MP3)

                    response = client.synthesize_speech(synthesis_input, voice, audio_config)

                    with open(imgname + '.mp3', 'wb') as out:
                        out.write(response.audio_content)
                        print('Audio content written to file "output.mp3"')

                    os.system('mpg321 -g 20 ' + imgname + '.mp3 &')
            except Exception as e:
                synthesis_input = texttospeech.types.SynthesisInput(text="No estoy seguro de lo que me muestras")
                voice = texttospeech.types.VoiceSelectionParams(
                    language_code='es-ES',
                    ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

                audio_config = texttospeech.types.AudioConfig(
                    audio_encoding=texttospeech.enums.AudioEncoding.MP3)

                response = client.synthesize_speech(synthesis_input, voice, audio_config)

                # The response's audio_content is binary.
                with open(imgname + '.mp3', 'wb') as out:
                    out.write(response.audio_content)

                os.system('mpg321 -g 20 ' + imgname + '.mp3 &')
                print("type error: " + str(e))


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

