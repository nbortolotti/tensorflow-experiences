import ConfigParser
import tensorflow as tf

config = ConfigParser.ConfigParser()
config.read('config.env')

keras_model = config.get('model', 'keras_model')
output_path = config.get('model', 'output_path')
input_model = config.get('model', 'input_model')

tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model(keras_model)
export_path = output_path

with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={input_model: model.input},
        outputs={t.name:t for t in model.outputs})