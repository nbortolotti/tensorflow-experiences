import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import cv2
from scipy.stats import itemfreq

MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'
# MODEL_NAME = 'faster_rcnn_inception_resnet_v2_atrous_coco_2017_11_08'

MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('object_detection/data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90

# Activate initial time to config the model
# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def dominant_color(np_object):
    pixs = np_object.reshape((-1, 3))

    n_colors = 5
    flags = cv2.KMEANS_RANDOM_CENTERS
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    _, labels, centroids = cv2.kmeans(np.float32(pixs), n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    dominant_object_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_object_color


def slice_box_objects(n_image, img_with, img_height, cla, b, sco, category="tie", confident=0.7):
    objects = []
    for i, box in enumerate(np.squeeze(b)):
        if category_index[np.squeeze(cla)[i]]['name'] == category:
            if np.squeeze(sco)[i] > confident:
                crop = n_image[int(box[0] * img_height):int(box[2] * img_height),
                       int(box[1] * img_with):int(box[3] * img_with)]
                plt.imshow(crop)
                plt.show()
                objects.append(crop)
    return objects


def object_detection():
    # Activate initial time to config the model

    # tar_file = tarfile.open(MODEL_FILE)
    # for file in tar_file.getmembers():
    #     file_name = os.path.basename(file.name)
    #     if 'frozen_inference_graph.pb' in file_name:
    #         tar_file.extract(file, os.getcwd())

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    PATH_TO_TEST_IMAGES_DIR = 'test_images'
    TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 2)]

    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)

                image_np_expanded = np.expand_dims(image_np, axis=0)

                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                (width, height) = image.size

                objs = slice_box_objects(image_np, width, height, classes, boxes, scores, "tie", 0.7)

                for o in objs:
                    print(dominant_color(o))


import argparse

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='explore objects images into image and detect dominant color ')
    pa.add_argument('--imagepath', dest='image_path', required=False, help='path of the image to analyze')

    args = pa.parse_args()
    # lista = args.schema.split(",")
    # demo = list(lista)

    object_detection()
    # result = xmlcsv(args.input_file, args.output_file, args.row, demo)
