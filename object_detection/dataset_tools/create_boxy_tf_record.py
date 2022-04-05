# Copyright 2022 Han's Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Convert raw Boxy dataset to TFRecord for object_detection.

This tool supports data generation for object detection (boxes, keypoint detection).

Please note that this tool creates sharded output files.

Example usage:
    python create_coco_tf_record.py  \
      --train_zip_dir="${TRAIN_ZIP_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_PATH/boxy_labels_train.json}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import json
import logging
import os
import zipfile

import cv2
import time

import contextlib2
import numpy as np
import tensorflow.compat.v1 as tf

from object_detection.dataset_tools import tf_record_creation_util
from object_detection.utils import dataset_util
from multiprocessing import Pool

flags = tf.app.flags
tf.flags.DEFINE_string('train_zip_dir', '', 'Training zip directory.')
tf.flags.DEFINE_string('train_annotations_file', '', 'Training annotations JSON file.')
tf.flags.DEFINE_string('boxy_label_scale', '0.5', 'Scale of Boxy scale.')
tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
tf.flags.DEFINE_string('num_shards', '10', "Number of Shards.")
tf.flags.DEFINE_string('num_pool_process', '10', "Number of multiprocess.")
tf.flags.DEFINE_string('num_accumulate_image_buffer', '1200', "Number of accumulate image buffer.")

FLAGS = flags.FLAGS

logger = tf.get_logger()
logger.setLevel(logging.INFO)

_BOXY_KEYPOINT_NAMES = [b'rear_lt', b'rear_rb', b'p0', b'p1', b'p2', b'p3']

FLAGS.boxy_label_scale = float(FLAGS.boxy_label_scale)
FLAGS.num_shards = int(FLAGS.num_shards)
FLAGS.num_pool_process = int(FLAGS.num_pool_process)
FLAGS.num_accumulate_image_buffer = int(FLAGS.num_accumulate_image_buffer)

def create_tf_example(key, img, label, scale=1.0):
  """Converts image and annotations to a tf.Example proto.

  Args:
    key: string of key from Boxy json label. Sample:"./bluefox_2016-09-30-15-19-35_bag/1475274353.721849.png"
    img: np.uint8 array RGB image
    label:list of dicts label of 'vehicles'
    scale: depend on the resolution of images from the zip file.
      0.5 for 1232x1028 pixels images
      1.0 fpr 2464x2056 pixels images

  Returns:
    key: SHA256 hash of the image.
    example: The converted tf.Example
  """
  image_height, image_width, _ = img.shape
  filename = key
  image_id = key

  encoded_jpg = tf.io.encode_jpeg(tf.constant(img))
  key = hashlib.sha256(encoded_jpg).hexdigest()

  xmin = []
  xmax = []
  ymin = []
  ymax = []
  is_crowd = []
  category_names = []
  category_ids = []
  area = []
  keypoints_x = []
  keypoints_y = []
  keypoints_visibility = []
  keypoints_name = []
  num_keypoints = []

  for l in label:
    bbox = l['AABB']
    xmin.append(bbox['x1'] * scale / image_width)
    xmax.append(bbox['x2'] * scale / image_width)
    ymin.append(bbox['y1'] * scale / image_height)
    ymax.append(bbox['y2'] * scale / image_height)
    is_crowd.append(0)
    category_ids.append(1)
    category_names.append(b"vehicle")
    area.append(0)

    # keypoint
    keypoints_name.extend(_BOXY_KEYPOINT_NAMES)

    num_kpts = 0
    if l['rear']:
      bbox = l['rear']
      keypoints_x.append(bbox['x1'] * scale / image_width)
      keypoints_x.append(bbox['x2'] * scale / image_width)
      keypoints_y.append(bbox['y1'] * scale / image_height)
      keypoints_y.append(bbox['y2'] * scale / image_height)
      keypoints_visibility.extend([2, 2])
      num_kpts += 2
    else:
      keypoints_x.extend([0.0]*2)
      keypoints_y.extend([0.0]*2)
      keypoints_visibility.extend([0, 0])

    if l['side']:
      p0 = l['side']['p0']
      p1 = l['side']['p1']
      p2 = l['side']['p2']
      p3 = l['side']['p3']
      keypoints_x.append(p0['x'] * scale / image_width)
      keypoints_x.append(p1['x'] * scale / image_width)
      keypoints_x.append(p2['x'] * scale / image_width)
      keypoints_x.append(p3['x'] * scale / image_width)
      keypoints_y.append(p0['y'] * scale / image_height)
      keypoints_y.append(p1['y'] * scale / image_height)
      keypoints_y.append(p2['y'] * scale / image_height)
      keypoints_y.append(p3['y'] * scale / image_height)
      keypoints_visibility.extend([2] * 4)
      num_kpts += 4
    else:
      keypoints_x.extend([0.0] * 4)
      keypoints_y.extend([0.0] * 4)
      keypoints_visibility.extend([0] * 4)
    num_keypoints.append(num_kpts)

  feature_dict = {
    'image/height': dataset_util.int64_feature(image_height),
    'image/width': dataset_util.int64_feature(image_width),
    'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
    'image/source_id': dataset_util.bytes_feature(image_id.encode('utf8')),
    'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
    'image/encoded': dataset_util.bytes_feature(encoded_jpg.numpy()),
    'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')),
    'image/object/bbox/xmin': dataset_util.float_list_feature(xmin),
    'image/object/bbox/xmax': dataset_util.float_list_feature(xmax),
    'image/object/bbox/ymin': dataset_util.float_list_feature(ymin),
    'image/object/bbox/ymax': dataset_util.float_list_feature(ymax),
    'image/object/class/text': dataset_util.bytes_list_feature(category_names),
    'image/object/is_crowd': dataset_util.int64_list_feature(is_crowd),
    'image/object/area': dataset_util.float_list_feature(area),
    'image/object/keypoint/x': (dataset_util.float_list_feature(keypoints_x)),
    'image/object/keypoint/y': (dataset_util.float_list_feature(keypoints_y)),
    'image/object/keypoint/num': (dataset_util.int64_list_feature(num_keypoints)),
    'image/object/keypoint/visibility': (dataset_util.int64_list_feature(keypoints_visibility)),
    'image/object/keypoint/text': (dataset_util.bytes_list_feature(keypoints_name))
    }

  example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
  return (key, example)

def _extract_rgb_image(key_name):
  _, zip_name, image_name = key_name.split("/")
  z_path = os.path.join(FLAGS.train_zip_dir, zip_name + ".zip")
  with zipfile.ZipFile(z_path, "r") as z:
    buf = z.read("{}/{}".format(zip_name, image_name))
    np_buf = np.frombuffer(buf, np.uint8)
    img = cv2.imdecode(np_buf, cv2.IMREAD_UNCHANGED)
    return img[:,:,::-1]

def _create_tf_record_from_boxy_annotations(annotations_file,
                                            output_path, num_shards, num_pool_process):
  """Loads Boxy annotation json file and converts to tf.Record format.

  Args:
    annotations_file: JSON file which stores the Boxy annotation.
    output_path: Path to output tf.Record file.
    num_shards: number of output file shards.
  """
  f = open(annotations_file)
  label_dict = json.load(f)
  with contextlib2.ExitStack() as tf_record_close_stack, \
      tf.gfile.GFile(annotations_file, 'r') as fid:
    output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
      tf_record_close_stack, output_path, num_shards)

    s = time.time()
    label_key_list = []
    cnt = 0

    for idx, label_key in enumerate(label_dict):

      if idx % 100 == 0:
        logging.info('On image %d of %d', idx, len(label_dict))
        print("time using : {} seconds".format(time.time()-s))
        s = time.time()

      def _extract_image_list(label_key_list):
        with Pool(num_pool_process) as p:
          img_list = p.map(_extract_rgb_image, label_key_list)
          return img_list

      def _create_tf_example_and_write_record(img_list, label_key_list, num_shards, output_tfrecords):
        for pool_idx, (label_key, img) in enumerate(zip(label_key_list, img_list)):
          label = label_dict[label_key]['vehicles']
          (_, tf_example) = create_tf_example(label_key, img, label, scale=FLAGS.boxy_label_scale)
          shard_idx = pool_idx % num_shards
          if tf_example:
            output_tfrecords[shard_idx].write(tf_example.SerializeToString())

      if cnt != FLAGS.num_accumulate_image_buffer:
        label_key_list.append(label_key)
        cnt += 1
      else:
        img_list = _extract_image_list(label_key_list)
        _create_tf_example_and_write_record(img_list, label_key_list, num_shards, output_tfrecords)
        label_key_list = []
        cnt = 0

    if len(label_key_list) != 0:
      img_list = _extract_image_list(label_key_list)
      _create_tf_example_and_write_record(img_list, label_key_list, num_shards, output_tfrecords)

def main(_):
    assert FLAGS.train_zip_dir, '`train_zip_dir` missing.'
    assert FLAGS.train_annotations_file, '`train_annotations_file` missing.'
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)
    train_output_path = os.path.join(FLAGS.output_dir, 'boxy_train.record')

    _create_tf_record_from_boxy_annotations(
        FLAGS.train_annotations_file,
        train_output_path,
        FLAGS.num_shards,
        FLAGS.num_pool_process
    )

if __name__ == '__main__':
  tf.app.run()
