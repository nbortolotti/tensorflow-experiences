"""
Usage:
  python od_video_label_detector.py

# validate dependencies todos
# validate fix parameters todos
"""

import sys
import os
import cv2
import numpy as np
import tensorflow as tf


sys.path.append("..")

from object_detection.utils import label_map_util # todo: dependency to custom_model_object_detection official model


MODEL_NAME = 'MODELFOLDER'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
PATH_TO_LABELS = os.path.join('LABELFOLDER', 'LABELFILE.pbtxt')

NUM_CLASSES = 1  # todo change to number of classes
IMAGE_SIZE = (12, 8)  # todo change to image size selected

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def detect_alert(boxes, classes, scores, category_index, max_boxes_to_draw=20,
                 min_score_thresh=.5,
                 ):
    r = []
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            test1 = None
            test2 = None

            if category_index[classes[i]]['name']:
                test1 = category_index[classes[i]]['name']
                test2 = int(100 * scores[i])

            line = {}
            line[test1] = test2
            r.append(line)

    return r


def detect_objects(image_np, sess, detection_graph):
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    alert_array = detect_alert(np.squeeze(boxes), np.squeeze(classes).astype(np.int32), np.squeeze(scores),
                               category_index)

    return alert_array


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


def process_image(image):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            alert_array = detect_objects(image, sess, detection_graph)

            alert = False

            for q in alert_array:
                print (q)
                if 'messi' in q: # todo: change label to detect
                    if q['messi'] > 10: # todo: change label confident 0-100 %
                        alert = True

            return alert


video = cv2.VideoCapture('INPUTVIDEO.mp4')  # todo: change video to analyze
success, image = video.read()
count = 0
success = True
while success:
    success, image = video.read()
    print 'Read a new frame: ', success
    if success:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        alert = process_image(img)
        if alert:
            break
    count += 1