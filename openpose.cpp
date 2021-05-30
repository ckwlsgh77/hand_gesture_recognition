#include <iostream>
#include <opencv2\opencv.hpp> 
#include <chrono>
#include <cmath>
#include <thread>
#include <Windows.h>

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

int recognition_gesture(std::vector<cv::Point> points);

par before_ret;
cv::Point cursor;
cv::Point before_point = { 0,0 };
cv::Mat frame;
cv::Mat result;
static bool is_thread_terminated;
static bool is_running;

int main() {

	cv::String protoFile = "openpose/models/hand/pose_deploy.prototxt";
	cv::String weightsFile = "openpose/models/hand/pose_iter_102000.caffemodel";

	cv::dnn::Net net = cv::dnn::readNetFromCaffe(protoFile, weightsFile);
	net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);

	cv::VideoCapture capture(0);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
	capture.set(cv::CAP_PROP_AUTOFOCUS, 0);
	if (!capture.isOpened()) {
		std::cout << "camera error" << std::endl;

		return 0;
	}

	is_running = false;
	std::thread t1(detector_thread, net); // 내부에서 while 등을 넣어서 loop.


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

				auto starting_point = std::chrono::system_clock::now();
				//detector_thread(net);
				current_state = HandGestureRecognition(net);
				std::this_thread::sleep_until(starting_point + kMinimumIntervalMs); //병렬실행을 위해 잠시 멈췄다감
			}

			if (current_state.check == 1) {
				DoAction(current_state);
			}

		}



		char key = cv::waitKey(24);
		if (key == 27) break;



	}

	is_thread_terminated = true;
	t1.join();



	return 0;
}

void detector_thread(cv::dnn::Net net) {
	while (!is_running) {
		Sleep(50);
	}

	while (!is_thread_terminated) {

		//std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

		cv::Mat clone_frame = frame.clone();

		int frameWidth = clone_frame.cols;
		int frameHeight = clone_frame.rows;

		cv::Mat inpBlob = cv::dnn::blobFromImage(clone_frame, 1.0 / 255, cv::Size(frameWidth, frameHeight), cv::Scalar(0, 0, 0), false, false); //이미지전처리 1/255 정규화

		net.setInput(inpBlob);

		cv::Mat output = net.forward();

		result = output.clone();

		//std::chrono::duration<double> sec = std::chrono::system_clock::now() - start;

		//std::cout << "시간(초) : " << sec.count() << " seconds" << std::endl; // release 0.16 debug 0.19

	}

}

int count = 0;
par HandGestureRecognition(cv::dnn::Net net) {
	int nPoints = 21;

	cv::Mat frameCopy = frame.clone();
	//cv::resize(frameCopy, frameCopy, cv::Size(620, 460));
	int H = result.size[2]; 
	int W = result.size[3];
	
	int frameWidth = frameCopy.cols;
	int frameHeight = frameCopy.rows;

	std::vector<cv::Point> points(nPoints);
	par ret;

	double sum_prob = 0;

	cv::Point maxLoc;
	double prob = 0.0;
	
	cv::String str = ".jpg";
	for (int n = 0; n < nPoints; n++)
	{

		cv::Mat probMap(H, W, CV_32F, result.ptr(0, n)); //n번째 keypoint의 heatmap
		//std::cout << *result.ptr(0, n) << std::endl;
		minMaxLoc(probMap, 0, &prob, 0, &maxLoc); //heatmap 최대값 위치


	//	if (probMap.cols > 0 && probMap.rows > 0) {

		//	cv::resize(probMap, probMap, cv::Size(480, 360));
		//	cv::namedWindow("main frame", cv::WINDOW_AUTOSIZE);
		//	imshow("heatmap", probMap);

		//}

		


		if (prob > 0.1)
		{
			circle(frameCopy, cv::Point((int)(maxLoc.x * 8), (int)(maxLoc.y * 8)), 4, cv::Scalar(0, 255, 255), -1);
			ret.points[n] = maxLoc;

		}


	}
	




	ret.count = recognition_gesture(ret.points);



	ret.check = 1;


	cv::flip(frameCopy, frameCopy, 1);

	if (ret.count)
		cv::putText(frameCopy, std::to_string(ret.count), cv::Point(10, 80), 0, 3, cv::Scalar(0, 255, 0), 3, 8, false);

	cv::resize(frameCopy, frameCopy, cv::Size(480, 360));
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

#define TWO_STEP_MOVEMENT_SPEED //천천히 움직이다가 클릭

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
#else // 손을 떼는순간 클릭
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



	}else{ // 일반

		double div_q = q / 10;

		if (div_q < 0)
			div_q = 1;	

		w = (double)p / (double)q; // 기울기
		m = -1 * (w*current_cursor.x - current_cursor.y); // y = wx + m --> m = -(wx-y)

		

		for (int i = 0; i < 9; i++) {
			tmp_cursor.x = tmp_cursor.x + (int)div_q;
			tmp_cursor.y = tmp_cursor.x * w + m;
			SetCursorPos(tmp_cursor.x, tmp_cursor.y);
		}
	}


	SetCursorPos(cur.x, cur.y);
}
