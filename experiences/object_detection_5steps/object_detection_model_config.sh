#!/usr/bin/env bash

git clone https://github.com/tensorflow/models.git
apt-get -qq install libprotobuf-java protobuf-compiler
protoc ./models/research/object_detection/protos/string_int_label_map.proto --python_out=.
cp -R models/research/object_detection/ object_detection/
cp -R models/research/slim/ slim/
rm -rf models