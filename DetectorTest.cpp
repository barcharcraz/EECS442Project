#include <cstdlib>
#include <string>
#include <vector>
#include <cstdio>
#include <chrono>
#include <opencv2/objdetect.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char** argv) {
	using namespace cv;
	using std::chrono::high_resolution_clock;
	using std::chrono::time_point;
	using std::chrono::duration;
	using std::chrono::milliseconds;
	using std::chrono::duration_cast;
	//using namespace std::literals;
	using std::vector;
	VideoCapture cap("VIRAT_S_000001.mp4");
	if (!cap.isOpened()) {
		std::exit(EXIT_FAILURE);
	}
	VideoWriter output;
	Size videoSize{ (int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) };
	output.open("output.mkv", VideoWriter::fourcc('D', 'I', 'V', 'X'), cap.get(CV_CAP_PROP_FPS), videoSize, true);
	auto hog = cuda::HOG::create();
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
	while (cap.read(img)) {
		start = high_resolution_clock::now();
		//cap >> img;
		if (!img.data) {
			continue;
		}
		vector<Rect> found;
		vector<Rect> found_filtered;
		orig = img;
		cvtColor(img, img_aux, COLOR_BGR2GRAY);
		resize(img_aux, img, Size(1280, 720));

		gpu_img.upload(img);

		//hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.02, 2);
		//hog.detect(img, found, 0, Size(8, 8), Size(32, 32));
		hog->detectMultiScale(gpu_img, found);
		auto ms = cap.get(CAP_PROP_POS_MSEC);
		auto s = ms / 1000;
		auto m = s / 60;
		ms = ms - int(s) * 1000;
		s = s - int(m) * 60;

		
		for (int i = 0; i < found.size(); ++i) {
			auto r = found[i];
			rectangle(orig, r.tl() * 1.5, r.br() * 1.5, cv::Scalar(0, 255, 0), 2);
		}
		if (found.size()) {
			output << orig;
		}
		end = high_resolution_clock::now();
		elapsed = end - start;
		fprintf(stderr, "Time: %d:%d:%d", (int)m, (int)s, (int)ms);
		fprintf(stderr, " MS/frame: %d ms\n", duration_cast<milliseconds>(elapsed).count());
		imshow("EECS442 Project", orig);
		if (waitKey(20) >= 0) {
			break;
		}
	}

	return EXIT_SUCCESS;
}