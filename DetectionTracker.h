#ifndef BARTOC_DETECTOR_TRACKING_H
#define BARTOC_DETECTOR_TRACKING_H
#include <opencv2/opencv.hpp>
#include <array>
template<typename T, int C>
class CircularBuffer {
  int _start = 0;
  int _size = 0;
  std::array<T, C> data;
public:
  
};
class DetectionTracker {
  const int track_size = 5;
public:
  DetectionTracker(int x, int y, double speed);
  void addDetection(const cv::Rect& detection);
  void addDetections(const std::vector<cv::Rect>& detections);
  int queryDetectionTrack(const cv::Rect& detection) const;
  std::vector<cv::Rect> getTrack(int i) const;
  const cv::Rect& getDetection(int i) const;
  int size() const;
  std::vector<cv::Rect> getTracks() const;
private:
  int winx;
  int winy;
  double eps_mult;
  std::vector<bool> updated_tracks;
  std::vector<cv::Rect> tracks;
  std::vector<int> track_times;
};

#endif
