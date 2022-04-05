import tensorflow as tf
import numpy as np
import cv2
from object_detection.core.preprocessor import preprocess
from object_detection.builders import preprocessor_builder
from object_detection.utils.config_util import get_configs_from_pipeline_file
from object_detection import read_from_tfrecord

tfrecord_path = "PATH OF TFRECORD"
image_path = "IMAGE PATH"
configure_path = "CONFIGURE FILE PATH"

dict_config_objs = get_configs_from_pipeline_file(configure_path)
data_augmentation_options = dict_config_objs['train_config'].data_augmentation_options
data_augmentation_options = [
        preprocessor_builder.build(step)
        for step in data_augmentation_options
    ]

parsed_image_dataset = read_from_tfrecord.decode_tfrecord(tfrecord_path)

for image_features in parsed_image_dataset:
  image_raw = tf.io.decode_jpeg(image_features['image/encoded']).numpy()
  xmin = image_features['image/object/bbox/xmin'].numpy()
  ymin = image_features['image/object/bbox/ymin'].numpy()
  xmax = image_features['image/object/bbox/xmax'].numpy()
  ymax = image_features['image/object/bbox/ymax'].numpy()

  keypoint_x = image_features['image/object/keypoint/x'].numpy()
  keypoint_y = image_features['image/object/keypoint/y'].numpy()

  groundtruth_boxes = np.concatenate(
      [
          np.expand_dims(ymin, 1),
          np.expand_dims(xmin, 1),
          np.expand_dims(ymax, 1),
          np.expand_dims(xmax, 1),
      ],
      1
  )
  keypoint_x = np.reshape(keypoint_x, (-1, 6))
  keypoint_y = np.reshape(keypoint_y, (-1, 6))
  groundtruth_keypoint = np.concatenate(
      [
          np.expand_dims(keypoint_y, -1),
          np.expand_dims(keypoint_x, -1),
      ],
      2
  )

  image_exp_dim = np.expand_dims(image_raw, 0)

  tensor_dict = {
      'image': tf.cast(tf.constant(image_exp_dim), tf.float32),
      'groundtruth_boxes': tf.constant(groundtruth_boxes),
      'groundtruth_classes': tf.constant([1]*len(groundtruth_boxes)),
      'groundtruth_weights': tf.constant([1]*len(groundtruth_boxes)),
      'groundtruth_keypoints':  tf.constant(groundtruth_keypoint)
  }

  output = preprocess(
      tensor_dict=tensor_dict,
      preprocess_options=data_augmentation_options,
  )
  output_image = output['image'].numpy()[0]
  frame_h, frame_w, _ = output_image.shape
  groundtruth_keypoint = output['groundtruth_keypoints'].numpy()
  px = [None] * 6
  py = [None] * 6
  for i , keypoint in enumerate(groundtruth_keypoint):
      for kpt_i, kpt_score in enumerate(keypoint):

          y, x = groundtruth_keypoint[i][kpt_i]
          x = int(x * frame_w)
          y = int(y * frame_h)
          cv2.circle(output_image, (x, y), 5, (255, 255, 0), -1)
          px[kpt_i] = x
          py[kpt_i] = y

      if px[0] and px[1]:
          cv2.rectangle(output_image, (px[0], py[0]), (px[1], py[1]), (0, 255, 255), 3, cv2.LINE_AA)
          cv2.circle(output_image, (px[0], py[0]), 5, (255, 0, 127), -1)
          cv2.circle(output_image, (px[1], py[1]), 5, (255, 0, 127), -1)
      if px[2] and px[3] and px[4] and px[5]:
          points = []
          for p_i in range(2, 6):
              points.append(np.array([px[p_i], py[p_i]]))
          points[2], points[3] = points[3], points[2]
          cv2.polylines(output_image, pts=[np.array(points)], isClosed=True, color=(0, 255, 255), thickness=3)

  output_image = np.uint8(output_image[:, :, ::-1])
  cv2.imshow('test', output_image)
  cv2.waitKey(0)

