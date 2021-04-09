#include<fstream>
#include<sstream>
#include<iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include<opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <thread>
#include <Windows.h>

using namespace std;
using namespace cv;
using namespace dnn;

typedef struct point_and_ret {
	std::vector<cv::Point> points;
	int count = 0;
	bool check = 0;
	int maxloc_n = 0;
	point_and_ret() {
		points.resize(22);
		count = 0;
	}

}par;

const int POSE_PAIRS[20][2] = {
	{ 0,1 },{ 1,2 },{ 2,3 },{ 3,4 },         // thumb
	{ 0,5 },{ 5,6 },{ 6,7 },{ 7,8 },         // pinkie
	{ 0,9 },{ 9,10 },{ 10,11 },{ 11,12 },    // middle
	{ 0,13 },{ 13,14 },{ 14,15 },{ 15,16 },  // ring
	{ 0,17 },{ 17,18 },{ 18,19 },{ 19,20 }   // small
};

par HandGestureRecognition(cv::dnn::Net net);
void DoAction(par action);
void detector_thread(cv::dnn::Net net);
void move_cursor(cv::Point cur);
void SendLeftDown();
void SendLeftUp();
int before_gesture = -1;
void remove_box(Mat&frame, const vector<Mat>&out);
// draw bounding boxes
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame);
// get output layers
vector<String> getOutputsNames(const Net& net);


int recognition_gesture(std::vector<cv::Point> points);

par before_ret;
cv::Point cursor;
cv::Point before_point = { 0,0 };
cv::Mat frame,hand;
cv::Mat blob;
static bool is_thread_terminated;
static bool is_running;
// confidence threshold
float conf_threshold = 0.5;
// nms threshold
float nms = 0.4;

int main() {

	cv::String protoFile = "openpose/models/hand/pose_deploy.prototxt";
	cv::String weightsFile = "openpose/models/hand/pose_iter_102000.caffemodel";

	// get labels of all classes
	cv::String yoloModel = "yolo/yolov3.cfg";
	cv::String yoloWeights = "yolo/yolov3_last.weights";

	cv::dnn::Net yolo_net = cv::dnn::readNetFromDarknet(yoloModel, yoloWeights);
	yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	cv::VideoCapture capture(0);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 416);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 416);
	capture.set(cv::CAP_PROP_AUTOFOCUS, 0);

	if (!capture.isOpened()) {
		std::cout << "camera error" << std::endl;

		return 0;
	}

	is_running = false;
	//std::thread t1(detector_thread, net); // 내부에서 while 등을 넣어서 loop.


	while (1)
	{

		POINT lp_cursor;

		GetCursorPos(&lp_cursor);

		cursor.x = lp_cursor.x;
		cursor.y = lp_cursor.y;

		static int count = 0;


		capture >> frame;

		if (frame.empty())
			break;
		else {

			par current_state;

			if (!is_running) {
				is_running = true;
				is_thread_terminated = false;
				Sleep(500);
			}
			else {
				constexpr std::chrono::milliseconds kMinimumIntervalMs(35);
				Mat hand_blob;
				blobFromImage(frame, hand_blob, 1.0 / 255, cv::Size(416, 416), Scalar(0, 0, 0), true, false);
				yolo_net.setInput(hand_blob);

				vector<Mat> outs;
				yolo_net.forward(outs, getOutputsNames(yolo_net));

				remove_box(frame, outs);

				if (hand.empty())
					continue;

				detector_thread(net);
				auto starting_point = std::chrono::system_clock::now();
				current_state = HandGestureRecognition(net);
				std::this_thread::sleep_until(starting_point + kMinimumIntervalMs);
			}

			if (current_state.check == 1) {
				DoAction(current_state);
			}

		}



		char key = cv::waitKey(24);
		if (key == 27) break;



	}

	is_thread_terminated = true;
	//t1.join();



	return 0;
}

void detector_thread(cv::dnn::Net net) {
	//while (!is_running) {
		//Sleep(50);
	//}

	//while (!is_thread_terminated) {

		std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

		cv::Mat clone_frame = hand.clone();

		int frameWidth = clone_frame.cols;
		int frameHeight = clone_frame.rows;

		cv::Mat inpBlob = cv::dnn::blobFromImage(clone_frame, 1.0 / 255, cv::Size(frameWidth, frameHeight), cv::Scalar(0, 0, 0), false, false);

		net.setInput(inpBlob);

		cv::Mat output = net.forward();

		blob = output.clone();

		std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;

		std::cout << "시간(초) : " << sec.count() << " seconds" << std::endl;

	//}

}

int count = 0;
par HandGestureRecognition(cv::dnn::Net net) {
	int nPoints = 21;

	cv::Mat frameCopy = frame.clone();
	//cv::resize(frameCopy, frameCopy, cv::Size(620, 460));
	int H = blob.size[2];
	int W = blob.size[3];

	int frameWidth = frameCopy.cols;
	int frameHeight = frameCopy.rows;

	std::vector<cv::Point> points(nPoints);
	par ret;

	double sum_prob = 0;

	cv::Point maxLoc;
	double prob = 0.0;

	for (int n = 0; n < nPoints; n++)
	{

		cv::Mat probMap(H, W, CV_32F, blob.ptr(0, n));

		minMaxLoc(probMap, 0, &prob, 0, &maxLoc);

		if (prob > 0.1)
		{
			circle(frameCopy, cv::Point((int)(maxLoc.x * 8), (int)(maxLoc.y * 8)), 4, cv::Scalar(0, 255, 255), -1);
			ret.points[n] = maxLoc;

		}


	}

	ret.count = recognition_gesture(ret.points);
	ret.check = 1;


	//cv::flip(frameCopy, frameCopy, 1);

	if (ret.count)
		cv::putText(frameCopy, std::to_string(ret.count), cv::Point(10, 80), 0, 3, cv::Scalar(0, 255, 0), 3, 8, false);

	cv::resize(frameCopy, frameCopy, cv::Size(416, 416));
	cv::namedWindow("main frame", cv::WINDOW_AUTOSIZE);
	imshow("main frame", frameCopy);


	return ret;
}

int recognition_gesture(std::vector<cv::Point> points) {

	bool left = 1;



	if (left) {
		if ((abs(points[0].x - points[4].x) + abs(points[0].y - points[4].y)) >
			(abs(points[0].x - points[3].x) + abs(points[0].y - points[3].y)) &&
			(abs(points[0].x - points[8].x) + abs(points[0].y - points[8].y)) >
			(abs(points[0].x - points[6].x) + abs(points[0].y - points[6].y)) &&
			(abs(points[0].x - points[12].x) + abs(points[0].y - points[12].y)) >
			(abs(points[0].x - points[11].x) + abs(points[0].y - points[11].y)) &&
			(abs(points[0].x - points[16].x) + abs(points[0].y - points[16].y)) >
			(abs(points[0].x - points[15].x) + abs(points[0].y - points[15].y)) &&
			(abs(points[0].x - points[20].x) + abs(points[0].y - points[20].y)) >
			(abs(points[0].x - points[19].x) + abs(points[0].y - points[19].y))) {

			if ((abs(points[8].x - points[4].x) + abs(points[8].y - points[4].y)) < 5)
			{

				return 4;
			}


			before_gesture = 1;
			return 1;
		}
		else if ((abs(points[0].x - points[8].x) + abs(points[0].y - points[8].y)) <
			(abs(points[0].x - points[5].x) + abs(points[0].y - points[5].y)) &&
			(abs(points[0].x - points[12].x) + abs(points[0].y - points[12].y)) <
			(abs(points[0].x - points[10].x) + abs(points[0].y - points[10].y)) &&
			(abs(points[0].x - points[16].x) + abs(points[0].y - points[16].y)) <
			(abs(points[0].x - points[14].x) + abs(points[0].y - points[14].y)) &&
			(abs(points[0].x - points[20].x) + abs(points[0].y - points[20].y)) <
			(abs(points[0].x - points[18].x) + abs(points[0].y - points[18].y))) {

			if ((abs(points[9].x - points[4].x) + abs(points[9].y - points[4].y)) <
				(abs(points[0].x - points[3].x) + abs(points[0].y - points[3].y))) {
				before_gesture = 2;
				return 2;
			}

		}
		else if ((abs(points[8].x - points[0].x) + abs(points[8].y - points[0].y)) <
			(abs(points[6].x - points[0].x) + abs(points[6].y - points[0].y)) &&
			((abs(points[0].x - points[4].x) + abs(points[0].y - points[4].y)) >
			(abs(points[0].x - points[3].x) + abs(points[0].y - points[3].y)) &&
				(abs(points[0].x - points[12].x) + abs(points[0].y - points[12].y)) >
				(abs(points[0].x - points[11].x) + abs(points[0].y - points[11].y)) &&
				(abs(points[0].x - points[16].x) + abs(points[0].y - points[16].y)) >
				(abs(points[0].x - points[15].x) + abs(points[0].y - points[15].y)) &&
				(abs(points[0].x - points[20].x) + abs(points[0].y - points[20].y)) >
				(abs(points[0].x - points[19].x) + abs(points[0].y - points[19].y)))) {

			return 3;
		}

		return before_gesture;
	}
	else {
		return before_gesture;
	}



	return before_gesture;
}

cv::Point current_point = { 0,0 };
cv::Point current_cursor;

int click_count = 0;
par before_state;

#define TWO_STEP_MOVEMENT_SPEED

void DoAction(par state) {
	if (before_state.count == 4 && click_count > 3 && state.count != 4) {
#ifdef TWO_STEP_MOVEMENT_SPEED 
		SendLeftDown();

		click_count = 0;


#endif
	}
	else if (state.count == 4) {
#ifdef TWO_STEP_MOVEMENT_SPEED 
		if (!(before_point.x == 0 && before_point.y == 0)) {

			cv::Point diff;
			diff.x = before_point.x - state.points[1].x;
			diff.y = before_point.y - state.points[1].y;

			if (abs(diff.x) <= 10 && abs(diff.y) <= 10) {

				cursor.x += diff.x * 15;
				cursor.y -= diff.y * 10;


				Sleep(10);
				move_cursor(cursor);
			}
		}
		click_count++;
#else
		if (click_count > 3) {
			SendLeftDown();

			click_count = 0;
		}
		else
		{
			click_count++;
		}
#endif
	}
	else if (state.count == 1) {
		SendLeftUp();
		click_count = 0;
	}
	else if (state.count == 2) {
		SendLeftUp();
		click_count = 0;
		if (!(before_point.x == 0 && before_point.y == 0)) {

			cv::Point diff;
			diff.x = before_point.x - state.points[1].x;
			diff.y = before_point.y - state.points[1].y;

			if (abs(diff.x) <= 10 && abs(diff.y) <= 10) {

				cursor.x += diff.x * 60;
				cursor.y -= diff.y * 40;


				Sleep(10);
				move_cursor(cursor);

			}
		}



	}
	else if (state.count == 3) {

		if (click_count > 5) {
			keybd_event(VK_BACK, 0, KEYEVENTF_EXTENDEDKEY, 0);
			Sleep(1);   //시간 1ms 지연
			keybd_event(VK_BACK, 0, KEYEVENTF_KEYUP, 0);

			click_count = 0;
			Sleep(1000);
		}
		else
		{
			click_count++;
		}
	}

	before_state = state;
	before_point.x = state.points[1].x;
	before_point.y = state.points[1].y;
}




void SendLeftDown() {


	INPUT input;
	ZeroMemory(&input, sizeof(INPUT));
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
	SendInput(1, &input, sizeof(INPUT));



}

void SendLeftUp() {


	INPUT input;
	ZeroMemory(&input, sizeof(INPUT));
	input.type = INPUT_MOUSE;
	input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
	SendInput(1, &input, sizeof(INPUT));


}




void move_cursor(cv::Point cur) {
#if 1
	POINT current_cursor;
	GetCursorPos(&current_cursor);

	POINT tmp_cursor = current_cursor;

	int q = cur.x - current_cursor.x;
	int p = cur.y - current_cursor.y;
	double w = 0;
	double m = 0;

	if (q == 0) { //x고정
		tmp_cursor.x = cur.x;

		double div_p = p / 10;

		if (div_p < 0)
			div_p = 1;



		for (int i = 0; i < 9; i++) {
			tmp_cursor.y = tmp_cursor.y + (int)div_p;
			SetCursorPos(tmp_cursor.x, tmp_cursor.y);
		}
	}

	else if (p == 0) { // y고정
		w = 0;
		tmp_cursor.y = cur.y;

		double div_q = q / 10;


		if (div_q < 0)
			div_q = 1;



		for (int i = 0; i < 9; i++) {
			tmp_cursor.x = tmp_cursor.x + (int)div_q;
			SetCursorPos(tmp_cursor.x, tmp_cursor.y);
		}



	}
	else { // 일반

		double div_q = q / 10;

		if (div_q < 0)
			div_q = 1;

		w = (double)p / (double)q;
		m = -1 * (w*current_cursor.x - current_cursor.y);



		for (int i = 0; i < 9; i++) {
			tmp_cursor.x = tmp_cursor.x + (int)div_q;
			tmp_cursor.y = tmp_cursor.x * w + m;
			SetCursorPos(tmp_cursor.x, tmp_cursor.y);
		}
	}



#endif 
	SetCursorPos(cur.x, cur.y);
}

void remove_box(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > conf_threshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	NMSBoxes(boxes, confidences, conf_threshold, nms, indices);

	int left_idx = 0;
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];

		if (boxes[left_idx].x > box.x) {
			left_idx = i;
		}

		//draw_box(classIds[idx], confidences[idx], box.x, box.y,
		//box.x + box.width, box.y + box.height, frame);
	}

	if (indices.size() > 0)
		draw_box(classIds[indices[left_idx]], confidences[indices[left_idx]], boxes[left_idx].x, boxes[left_idx].y,
			boxes[left_idx].x + boxes[left_idx].width, boxes[left_idx].y + boxes[left_idx].height, frame);
}

// Draw the predicted bounding box
void draw_box(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	if (bottom > frame.rows)
		bottom = frame.rows;
	if (top < 0)
		top = 0;
	if (left < 0)
		left = 0;
	if (right > frame.cols)
		right = frame.cols;
	hand = frame(Range(top, bottom), Range(left, right));
	imshow("hand", hand);

}

vector<String> getOutputsNames(const Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}