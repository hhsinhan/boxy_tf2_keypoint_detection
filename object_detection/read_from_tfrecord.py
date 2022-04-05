import tensorflow as tf

# Create a dictionary describing the features.
image_feature_description = {
    'image/height': tf.io.FixedLenFeature([], tf.int64),
    'image/width': tf.io.FixedLenFeature([], tf.int64),
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/source_id': tf.io.FixedLenFeature([], tf.string),
    'image/key/sha256': tf.io.FixedLenFeature([], tf.string),
    'image/encoded': tf.io.FixedLenFeature([], tf.string),
    'image/format': tf.io.FixedLenFeature([], tf.string),
    'image/object/bbox/xmin': tf.io.RaggedFeature(tf.float32),
    'image/object/bbox/ymin': tf.io.RaggedFeature(tf.float32),
    'image/object/bbox/xmax': tf.io.RaggedFeature(tf.float32),
    'image/object/bbox/ymax': tf.io.RaggedFeature(tf.float32),
    'image/object/class/text': tf.io.RaggedFeature(tf.string),
    'image/object/is_crowd': tf.io.RaggedFeature(tf.int64),
    'image/object/area': tf.io.RaggedFeature(tf.float32),
    'image/object/keypoint/x': tf.io.RaggedFeature(tf.float32),
    'image/object/keypoint/y': tf.io.RaggedFeature(tf.float32),
    'image/object/keypoint/num': tf.io.RaggedFeature(tf.int64),
    'image/object/keypoint/visibility': tf.io.RaggedFeature(tf.int64),
    'image/object/keypoint/text': tf.io.RaggedFeature(tf.string),

}

def _parse_image_function(example_proto):
  return tf.io.parse_single_example(example_proto, image_feature_description)

def decode_tfrecord(tfrecord_path):
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_path)
    return raw_image_dataset.map(_parse_image_function)


