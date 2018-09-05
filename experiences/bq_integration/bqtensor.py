import tensorflow as tf
import pandas as pd
import ConfigParser

tf.enable_eager_execution()  # eager

config = ConfigParser.ConfigParser()
config.read('config.env')

project_id = config.get('google','cloud_id') #is needed use a cloud project id
df_train = pd.io.gbq.read_gbq('''SELECT * FROM [socialagilelearning:iris.training]''', project_id=project_id, private_key=config.get('google','service_key'), verbose=False)
df_test = pd.io.gbq.read_gbq('''SELECT * FROM [socialagilelearning:iris.test]''', project_id=project_id, private_key=config.get('google','service_key'), verbose=False)


categories='Plants'
train_plantfeatures, train_categories = df_train, df_train.pop(categories)
test_plantfeatures, test_categories = df_test, df_test.pop(categories)

y_categorical = tf.contrib.keras.utils.to_categorical(train_categories, num_classes=3)
y_categorical_test = tf.contrib.keras.utils.to_categorical(test_categories, num_classes=3)

dataset = tf.data.Dataset.from_tensor_slices((train_plantfeatures.values, y_categorical))
dataset = dataset.batch(32)
dataset = dataset.shuffle(1000)
dataset = dataset.repeat()


dataset_test = tf.data.Dataset.from_tensor_slices((test_plantfeatures.values, y_categorical_test))
dataset_test = dataset_test.batch(32)
dataset_test = dataset_test.shuffle(1000)
dataset_test = dataset_test.repeat()

model = tf.keras.Sequential([
  tf.keras.layers.Dense(16, input_dim=4),
  tf.keras.layers.Dense(3, activation=tf.nn.softmax),
])

opt = tf.train.GradientDescentOptimizer(learning_rate=0.001)
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(dataset, steps_per_epoch=32, epochs=100, verbose=1)

loss, accuracy = model.evaluate(dataset_test, steps=32)

print("loss:%f"% (loss))
print("accuracy: %f"%   (accuracy))