# Boxy Keypoints Detection with C++ Code

Adapting the code from tensorflow2_cpp (https://github.com/borarak/tensorflow2_cpp).
Sample code import saved_model from tensorflow 2 serve prediction in C++

## 0. Install Environment
### Step 1. Insall tensorflow2_cpp 
clone tensorflow2_cpp (https://github.com/borarak/tensorflow2_cpp) and build the docker envirnment.


### Step 2. Compile Source 
1. Replace "tensorflow2_cpp/get_prediction.cpp" and "tensorflow2_cpp/saved_model_loader.h" into "boxy_tf2_keypoint_detection/cpp_code/get_prediction.cpp" and "boxy_tf2_keypoint_detection/cpp_code/saved_model_loader.h" 
2. Start container and mount the model volume
```bash
docker run --gpus all -it --rm -v PATH_OF_SAVED_MODEL/:/object_detection/models/ boraraktim/tensorflow2_cpp
```
3. build the code
```bash
mkdir build
cd build 
cmake ..
make
```
### Step 3. Predict
```
./get_prediction <path/to/saved_model> <path/to/image.jpg> <path/to/output.jpg>
```
