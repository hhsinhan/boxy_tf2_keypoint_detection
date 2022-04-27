/***
 * This is a sample program to demonstrate the use of the model_loader.h files
 ***/

#include "./saved_model_loader.h"
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#define THRESHOLD (0.4)
#define KP_THRESHOLD (0.35)


int main(int argc, char* argv[]){
	if (argc != 4){
		std::cout << "Error! Usage: <path/to_saved_model> <path/to_input/image.jpg> <path/to/output/image.jpg>" << std::endl;
		return 1;
	}

	// Make a Prediction instance
	Prediction out_pred;
	out_pred.boxes = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
	out_pred.scores = std::unique_ptr<std::vector<float>>(new std::vector<float>());
	out_pred.labels = std::unique_ptr<std::vector<int>>(new std::vector<int>());
	out_pred.kps_scores = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());
	out_pred.keypoints = std::unique_ptr<std::vector<std::vector<float>>>(new std::vector<std::vector<float>>());

	const string model_path = argv[1]; 
	const string test_image_file  = argv[2];
	const string test_prediction_image = argv[3];

	// Load the saved_model
	ModelLoader model(model_path);

	//Predict on the input image
	model.predict(test_image_file, out_pred);

	using namespace cv;
	Mat img = imread(test_image_file, IMREAD_COLOR);

	Size size = img.size();
	int height = size.height;
	int width = size.width;

	auto boxes = (*out_pred.boxes);
	auto scores = (*out_pred.scores);
	auto kps_scores = (*out_pred.kps_scores);
	auto keypoints = (*out_pred.keypoints);


	for (int i=0; i < boxes.size(); i++){
	    auto box = boxes[i];
	    auto score = scores[i];
	    if (score < THRESHOLD){
	        continue;
	    }
		int ymin = (int) (box[0] * height);
		int xmin = (int) (box[1] * width);
		int h = (int) (box[2] * height) - ymin;
		int w = (int) (box[3] * width) - xmin;
		Rect rect = Rect(xmin, ymin, w, h);
		rectangle(img, rect, cv::Scalar(0, 0, 255), 2);
		auto kp_scores = kps_scores[i];
		auto keypoint = keypoints[i];
		Point rear_lt(-1, -1);
		Point rear_rb(-1, -1);
		Point p0(-1, -1);
		Point p1(-1, -1);
		Point p2(-1, -1);
		Point p3(-1, -1);
		for (int j = 0; j < kp_scores.size(); j++){

			int x = j*2+1;
			int y = j*2;

			if (kp_scores[j] > KP_THRESHOLD){
				switch (j)
				{
				case 0:
					rear_lt.x = (int) (keypoint[x] * width);
					rear_lt.y = (int) (keypoint[y] * height);
					break;
				case 1:
					rear_rb.x = (int) (keypoint[x] * width);
					rear_rb.y = (int) (keypoint[y] * height);
					break;
				case 2:
					p0.x = (int) (keypoint[x] * width);
					p0.y = (int) (keypoint[y] * height);
					break;
				case 3:
					p1.x = (int) (keypoint[x] * width);
					p1.y = (int) (keypoint[y] * height);
					break;
				case 4:
					p2.x = (int) (keypoint[x] * width);
					p2.y = (int) (keypoint[y] * height);
					break;
				case 5:
					p3.x = (int) (keypoint[x] * width);
					p3.y = (int) (keypoint[y] * height);
					break;
				}
			}
		}
		if (rear_lt.x >= 0 && rear_rb.x >= 0)
			rectangle(img, cv::Rect(rear_lt, rear_rb), cv::Scalar(0, 255, 255), 2);
		if (p0.x >= 0 && p1.x >= 0 && p2.x >= 0 && p3.x >= 0){
			std::vector<cv::Point > contour = { p0, p1, p3, p2 };
			std::vector<std::vector<cv::Point >> contours;
			contours.push_back(contour);
			cv::polylines(img, contours, true, cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
		}
	}

	if (img.empty()){
		std::cout <<" Failed to read image" << std::endl;
	}

	imwrite(test_prediction_image, img);
}
