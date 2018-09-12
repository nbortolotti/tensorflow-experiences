import tensorflow as tf

tf.keras.backend.set_learning_phase(0)
model = tf.keras.models.load_model('./iris_model.h5')
export_path = './iris/2'

with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'input_iris': model.input},
        outputs={t.name:t for t in model.outputs})