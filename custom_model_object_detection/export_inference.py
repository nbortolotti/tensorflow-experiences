"""
Usage:
  python export_inference.py

# validate dependencies todos
# validate fix parameters todos
"""

import tensorflow as tf
from google.protobuf import text_format
from object_detection import exporter  # todo: dependency to custom_model_object_detection official model
from object_detection.protos import pipeline_pb2  # todo: dependency to custom_model_object_detection official model


def main(_):
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile('FILE.config', 'r') as f:  # todo: change config file
        text_format.Merge(f.read(), pipeline_config)
    exporter.export_inference_graph(
        'image_tensor', pipeline_config, 'FOLDER/MODEL.ckpt-NUMBER', 'OUTPUT_FOLDER')  # todo: model checkpoint, output folder
