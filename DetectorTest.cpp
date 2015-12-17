#include <cstdlib>
#include <string>
#include <vector>
#include <cstdio>
#include <chrono>
#include <string>
#include <opencv2/objdetect.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include "DetectionTracker.h"

int main(int argc, char** argv) {
	using namespace cv;
	using std::chrono::high_resolution_clock;
	using std::chrono::time_point;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	using std::chrono::duration_cast;
	using std::to_string;
	//using namespace std::literals;
	using std::vector;
	const int frameskip = 1;
	VideoCapture cap("VIRAT_S_000001.mp4");
	if (!cap.isOpened()) {
		fprintf(stderr, "Could not open video\n");
		std::exit(EXIT_FAILURE);
	}
	VideoWriter output;
	Size videoSize{ (int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) };
	output.open("output.mkv", VideoWriter::fourcc('D', 'I', 'V', 'X'), cap.get(CV_CAP_PROP_FPS), videoSize, true);
	auto hog = cuda::HOG::create();
	//auto hog = cv::Ptr<cv::HOGDescriptor>(new cv::HOGDescriptor());
	Mat detector = hog->getDefaultPeopleDetector();
	hog->setSVMDetector(detector);
	hog->setNumLevels(9);
	hog->setHitThreshold(0);
	hog->setScaleFactor(1.02);
	hog->setGroupThreshold(2);

	namedWindow("EECS442 Project", CV_WINDOW_AUTOSIZE);
	Mat img;
	Mat img_aux;
	Mat orig;
	cuda::GpuMat gpu_img;
	// set up clocks to time our function
	time_point<high_resolution_clock> start, end;
	duration<double> elapsed;
	DetectionTracker tracker(1280, 720, frameskip);
	int lastTracks;
	vector<Rect> found;
	while (cap.read(img)) {
		start = high_resolution_clock::now();
		//cap >> img;
		if (!img.data) {
			continue;
		}
		long long frame = cap.get(CV_CAP_PROP_POS_FRAMES);
		orig = img;
		cvtColor(img, img_aux, COLOR_BGR2GRAY);
		resize(img_aux, img, Size(1280, 720));
		if (!(frame % frameskip)) {
			gpu_img.upload(img);
			found = vector<Rect>{};
			assert(found.size() == 0);
			//hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.02, 2);
			//hog.detect(img, found, 0, Size(8, 8), Size(32, 32));
			hog->detectMultiScale(gpu_img, found);
			assert(found.size() < 150);
			auto ms = cap.get(CAP_PROP_POS_MSEC);
			auto s = ms / 1000;
			auto m = s / 60;
			ms = ms - int(s) * 1000;
			s = s - int(m) * 60;


			tracker.addDetections(found);
			end = high_resolution_clock::now();
			elapsed = end - start;
			fprintf(stderr, "Time: %d:%d:%d", (int)m, (int)s, (int)ms);
			fprintf(stderr, " MS/frame: %d ms\n", duration_cast<milliseconds>(elapsed).count());
		}
		for (int i = 0; i < tracker.size(); ++i) {
			auto& r = tracker.getDetection(i);
			putText(orig, to_string(i), r.tl() * 1.5, FONT_HERSHEY_PLAIN, 2, Scalar(0, 0, 255, 255));
			rectangle(orig, r.tl() * 1.5, r.br() * 1.5, cv::Scalar(0, 255, 0), 2);

		}

		if (found.size() || lastTracks != tracker.size()) {
			output << orig;
		}
		lastTracks = tracker.size();
		imshow("EECS442 Project", orig);
		if (waitKey(20) >= 0) {
			break;
		}
	}

	return EXIT_SUCCESS;
}
