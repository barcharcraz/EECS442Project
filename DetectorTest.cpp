#include <cstdlib>
#include <string>
#include <vector>
#include <cstdio>
#include <opencv2/objdetect.hpp>
#include <opencv2/cudaobjdetect.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

int main(int argc, char** argv) {
  using namespace cv;
  //using namespace std::literals;
  using std::vector;
  VideoCapture cap("VIRAT_S_000001.mp4");
  if (!cap.isOpened()) {
    std::exit(EXIT_FAILURE);
  }
  VideoWriter output;
  Size videoSize{ (int)cap.get(CV_CAP_PROP_FRAME_WIDTH), (int)cap.get(CV_CAP_PROP_FRAME_HEIGHT) };
  output.open("output.mkv", VideoWriter::fourcc('D','I','V','X'), cap.get(CV_CAP_PROP_FPS), videoSize, true);
  cap.set(CAP_PROP_POS_MSEC, (2 * 60 + 47) * 1000);
  auto hog = cuda::HOG::create();
  Mat detector = hog->getDefaultPeopleDetector();
  hog->setSVMDetector(detector);
  
  namedWindow("EECS442 Project", CV_WINDOW_AUTOSIZE);
  Mat img;
  while (true) {
    cap >> img;
    if (!img.data) {
      continue;
    }
    vector<Rect> found;
    vector<Rect> found_filtered;
    //hog.detectMultiScale(img, found, 0, Size(8, 8), Size(32, 32), 1.02, 2);
    //hog.detect(img, found, 0, Size(8, 8), Size(32, 32));
	hog->detectMultiScale(img, found);
    fprintf(stderr, "Size: %d", found.size());
    for (int i = 0; i < found.size(); ++i) {
      auto r = found[i];
      rectangle(img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
    }
    if (found.size()) {
      output << img;
    }
    imshow("EECS442 Project", img);
    if (waitKey(20) >= 0) {
      break;
    }
  }
  
  return EXIT_SUCCESS;
}