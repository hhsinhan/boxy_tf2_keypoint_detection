import tensorflow as tf
import cv2
import numpy as np
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string('model_path', None, 'PATH OF MODEL')
flags.DEFINE_string('image_path', None, 'PATH OF IMAGE')

def _image_detection():
    model_path = FLAGS.model_path
    image_path = FLAGS.image_path

    loaded = tf.saved_model.load(model_path)
    infer = loaded.signatures["serving_default"]

    frame = cv2.imread(image_path)

    frame_exp = np.expand_dims(frame,0)
    detection_result = infer(tf.constant(frame_exp))

    detection_scores = detection_result['detection_scores'].numpy()[0]
    detection_boxes = detection_result['detection_boxes'].numpy()[0]
    detection_keypoints = detection_result['detection_keypoints'].numpy()[0]
    detection_keypoint_scores = detection_result['detection_keypoint_scores'].numpy()[0]

    frame_h , frame_w, _ = frame.shape
    for i, score in enumerate(detection_scores):
        if score > 0.5:
            ymin, xmin, ymax, xmax = detection_boxes[i]
            ymin = int(ymin * frame_h)
            xmin = int(xmin * frame_w)
            ymax = int(ymax * frame_h)
            xmax = int(xmax * frame_w)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,0,255), 3, cv2.LINE_AA)

            for kpt_i, kpt_score in enumerate(detection_keypoint_scores[i]):
                if kpt_score > 0.3:
                    y , x = detection_keypoints[i][kpt_i]
                    x = int(x * frame_w)
                    y = int(y * frame_h)
                    cv2.circle(frame, (x,y), 5, (255,255,0), -1)

    cv2.imshow('frame', frame)
    cv2.waitKey(0)

def main(argv):
    flags.mark_flag_as_required('model_path')
    flags.mark_flag_as_required('image_path')
    _image_detection()


if __name__ == '__main__':
    tf.compat.v1.app.run()
