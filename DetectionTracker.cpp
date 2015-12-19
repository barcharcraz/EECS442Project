#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "DetectionTracker.h"
using cv::Rect;
using std::vector;
using std::sqrt;
using std::numeric_limits;
static double dist(const Rect& a, const Rect& b) {
    auto a_center = (a.tl() + a.br()) / 2.0;
    auto b_center = (b.tl() + b.br()) / 2.0;
    auto diff = a_center - b_center;
    return sqrt(diff.dot(diff));
}
DetectionTracker::DetectionTracker(int x, int y, double speed)
    : winx(x), winy(y), eps_mult(speed)
{}
void DetectionTracker::addDetection(const Rect& detection) {
    const double eps = 0.5;
    int idx = queryDetectionTrack(detection);
    if(idx == -1) {
        //if we were too far away from any track to match
        //then insert the rectangle in a new track
        tracks.push_back(detection);
        track_times.push_back(0);
        updated_tracks.push_back(true);
    } else {
        if(updated_tracks[idx]) {
            //we have already updated the closest track
            //TODO: deal with this once we are using a ring buffer

        }
        //otherwise replace the current track rectangle with
        //the new detection
        tracks[idx] = detection;
        track_times[idx] = 0;
        updated_tracks[idx] = true;
    }

}
void DetectionTracker::addDetections(const vector<Rect>& detections) {
	for (auto& b : updated_tracks) b = false;
    for(auto& elm : track_times) elm += 1;
    for(auto& elm : detections) {
        addDetection(elm);
    }
	assert(tracks.size() == updated_tracks.size());
	assert(tracks.size() == track_times.size());
    for(int i = 0; i < tracks.size(); ++i) {
		assert(tracks.size() == updated_tracks.size());
		assert(tracks.size() == track_times.size());
        const double time_limit = 15.0 / eps_mult;
        fprintf(stderr, "%d: %f", i, double(track_times[i]));
        if(double(track_times[i]) > time_limit) {
            tracks.erase(begin(tracks)+i);
            updated_tracks.erase(begin(updated_tracks)+i);
            track_times.erase(begin(track_times)+i);
        }
    }
}
int DetectionTracker::queryDetectionTrack(const Rect& detection) const {
    const double eps = 10 * eps_mult;
    double mindist = numeric_limits<double>::infinity();
    int minidx = -1;
    for(int i = 0; i < tracks.size(); ++i) {
        double curdist = dist(tracks[i], detection);
        if(curdist < eps && curdist < mindist) {
            mindist = curdist;
            minidx = i;
        }
    }
    return minidx;
}
vector<Rect> DetectionTracker::getTrack(int i) const {
    return vector<Rect>{tracks[i]};
}
vector<Rect> DetectionTracker::getTracks() const {
    return tracks;
}
const Rect& DetectionTracker::getDetection(int i) const {
    return tracks[i];
}
int DetectionTracker::size() const {
    return tracks.size();
}
