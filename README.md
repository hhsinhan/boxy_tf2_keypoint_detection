# Boxy Keypoints Detection 

This project used Boxy dataset from Bosch and Centernet from the Tensorflow Object Detection API to detect vehcles 3D box.

## 0. Install Environment
### Step 1. Install Bosch Boxy Dataset
Download Dataset From : https://boxy-dataset.com/boxy/
### Step 2. Install Object Detection API
Install Tensorflow Object Detection API : https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2.md
### Step 3. Clone and cover 
Clone this project and cover and overwrite the direction to object_detection
### Step 4. Update protos
```
cd models/research
# Update protos.
protoc object_detection/protos/*.proto --python_out=.
```
## 1. Train Boxy
### Step 1. Create Training Dataset
 
```
cd object_detection/dataset_tools/
python create_coco_tf_record.py  \
      --train_zip_dir="${TRAIN_ZIP_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_PATH/boxy_labels_train.json}" \
      --output_dir="${OUTPUT_DIR}"
```
### Step 2. Modify TF record file in Configure file
Can pick a configure file for training
```
object_detection/configs/tf2/centernet_mobilenet_v2_fpn_512x512_kpts_boxy.config
object_detection/configs/tf2/centernet_resnet50_v2_512x512_kpts_boxy.config
```
Edited the path of the tfrecord path into the configure file you selected.
```
tf_record_input_reader {
    input_path: MODIFY_HERE
  }
```
### Step 3. Train Model
Train the model by follewing command:
```
python model_main_tf2.py \
  --model_dir MODEL_DIR \
  --pipeline_config_path PIPELINE_CONFIG_PATH
```
## 2. Export SavedModel
Export to savemodel pb file from checkpoint:
```
python exporter_main_v2.py \
    --pipeline_config_path PIPELINE_CONFIG_PATH \
    --trained_checkpoint_dir CHECKPOINT_DIR \
    --output_directory OUTPUT_DIR 
```

## 3. Inference Model
### Inference model on image
``` 
python object_detection/detection_kpt_by_image.py \
    --model_path PATH_OF_MODEL \
    --image_path PATH_OF_VIDEO 
```
### Inference model on video
```
python object_detection/detection_kpt_by_video.py \
    --model_path PATH_OF_MODEL \
    --video_path PATH_OF_VIDEO 
```
